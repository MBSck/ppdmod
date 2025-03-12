from functools import partial

import astropy.units as u
import numpy as np
from astropy.modeling.models import BlackBody
from scipy.interpolate import interp1d
from scipy.special import j0, jv

from .base import Component, FourierComponent
from .options import OPTIONS
from .parameter import Parameter
from .utils import compare_angles


class NBandFit(Component):
    name = "NBandFit"
    description = "A fit to the SED of a star."
    label = "NBandFit"

    def __init__(self, **kwargs):
        """The class's constructor."""
        super().__init__(**kwargs)
        self.tempc = Parameter(base="tempc")
        self.pah = Parameter(base="pah")
        self.scale_pah = Parameter(base="scale_pah")
        self.f = Parameter(
            name="f",
            unit=u.one,
            description="Offset",
            free=True,
            base="f",
        )

        self.materials = list(
            ["_".join(key.split("_")[1:]) for key in kwargs.keys() if "kappa" in key]
        )

        for material in self.materials:
            for prefix in ["kappa", "weight"]:
                key = "kappa_abs" if prefix == "kappa" else "weight_cont"
                param_name = f"{prefix}_{material}"
                param = Parameter(
                    name=param_name,
                    description=f"The mass fraction for {param_name}",
                    base=key,
                )

                setattr(self, param_name, param)

        self.eval(**kwargs)

    def flux_func(self, wavelength: u.um) -> np.ndarray:
        """Returns the flux weight of the point source."""
        bb = BlackBody(temperature=self.tempc())(wavelength)

        # NOTE: The 1e2 term is to be able to fit the weights as percentages
        opacity = np.sum(
            [
                getattr(self, f"weight_{material}")().value
                / 1e2
                * getattr(self, f"kappa_{material}")(wavelength)
                for material in self.materials
            ],
            axis=0,
        )

        pah = self.scale_pah() * self.pah(wavelength)
        flux = (bb * opacity * u.sr * 10.0 ** -self.f().value).to(u.Jy) + pah
        return flux.value.reshape((wavelength.size, 1))


class Point(FourierComponent):
    """Point source."""

    name = "Point"
    description = "Point source."

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dist = Parameter(base="dist")
        self.eval(**kwargs)

    def flux_func(self, wl: u.um) -> np.ndarray:
        """Computes the flux of the star."""
        fr = self.fr(wl).value
        if not isinstance(fr, (tuple, list, np.ndarray)):
            fr = np.array([fr])[:, np.newaxis]
        else:
            fr = fr.reshape((wl.size, 1))
        return fr

    def vis_func(self, spf: 1 / u.rad, psi: u.rad, wl: u.um, **kwargs) -> np.ndarray:
        """Computes the complex visibility."""
        vis = np.zeros_like(spf).value
        vis[:] = self.flux_func(wl)[(...,) + (len(vis.shape[1:]) - 1) * (np.newaxis,)]
        return vis.astype(OPTIONS.data.dtype.complex)

    def image_func(
        self, xx: u.mas, yy: u.mas, pixel_size: u.mas, wl: u.m = None
    ) -> np.ndarray:
        """Computes the image from a 2D grid."""
        image = np.zeros((wl.size, *xx.shape))
        rho = np.hypot(xx, yy)
        y_ind, x_ind = np.unravel_index(np.argmin(rho), xx.shape)
        star_flux = (self.compute_flux(wl) / 4)[..., np.newaxis]
        image[:, y_ind - 1 : y_ind + 1, x_ind - 1 : x_ind + 1] = star_flux
        return image


class Ring(FourierComponent):
    """A ring.

    Parameters
    ----------
    rin : astropy.units.mas
        The inner radius of the ring
    thin : bool
        If toggled the ring is infinitesimal. Default is 'True'.
    rout : astropy.units.mas
        The outer radius of the ring. Applies only for 'False' thin
    """

    name = "Ring"
    description = "A simple ring."
    thin = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rin = Parameter(base="rin")
        self.rout = Parameter(base="rout")

        self.eval(**kwargs)

    def compute_internal_grid(self) -> u.Quantity:
        """Computes the model grid.

        Returns
        -------
        radial_grid
        """
        dim = self.rin.value, self.dim.value
        rin, rout = self.rin.value, self.rout.value
        dim, dtype = self.dim.value, OPTIONS.data.dtype.real
        if OPTIONS.model.gridtype == "linear":
            return np.linspace(rin, rout, dim).astype(dtype) * self.rin.unit
        return (
            np.logspace(np.log10(rin), np.log10(rout), dim).astype(dtype)
            * self.rin.unit
        )

    def vis_func(self, spf: 1 / u.rad, psi: u.rad, wl: u.um, **kwargs) -> np.ndarray:
        """Computes the complex visibility."""
        mod_amps, cos_diff, bessel_funcs = [], [], []
        if self.asymmetric:
            for i in range(1, OPTIONS.model.modulation + 1):
                rho, theta = getattr(self, f"rho{i}")(), getattr(self, f"theta{i}")()
                mod_amps.append((-1j) ** i * rho)
                cos_diff.append(
                    np.cos(i * compare_angles(psi.value, theta.to(u.rad).value))
                )
                bessel_funcs.append(partial(jv, i))

            mod_amps = np.array(mod_amps)
            cos_diff = np.array(cos_diff)
            bessel_funcs = np.array(bessel_funcs)

        def _vis_func(xx: np.ndarray):
            """Shorthand for the vis calculation."""
            nonlocal mod_amps, cos_diff, bessel_funcs

            vis = j0(xx).astype(complex)
            if self.asymmetric:
                bessel_funcs = np.array(list(map(lambda x: x(xx), bessel_funcs)))
                mod_amps = mod_amps.reshape(
                    (mod_amps.shape[0],) + (1,) * (bessel_funcs.ndim - 1)
                )
                vis += (mod_amps * cos_diff * bessel_funcs).sum(axis=0)
            return vis

        if self.thin:
            vis = _vis_func(2 * np.pi * self.rin().to(u.rad) * spf)
        else:
            intensity_func = kwargs.pop("intensity_func", None)
            radius = self.compute_internal_grid()

            if intensity_func is not None:
                intensity = intensity_func(radius, wl).to(
                    u.erg / (u.rad**2 * u.cm**2 * u.s * u.Hz)
                )
                intensity = intensity[:, np.newaxis]

            if radius.unit not in [u.rad, u.mas]:
                radius = (radius.to(u.au) / self.dist().to(u.pc)).value * 1e3 * u.mas

            radius = radius.to(u.rad)
            vis = _vis_func(2 * np.pi * radius * spf)
            if intensity_func is None:
                vis = np.trapezoid(vis, radius)
            else:
                if len(vis.shape) >= 4:
                    intensity = intensity[..., np.newaxis, :]
                vis = np.trapezoid(radius * intensity * vis, radius).to(u.Jy)

            if intensity_func is None:
                vis /= (self.rout() - self.rin()).to(u.rad)
            else:
                vis *= 2 * np.pi * self.cinc()

        return vis.value.astype(OPTIONS.data.dtype.complex)

    def image_func(
        self, xx: u.mas, yy: u.mas, pixel_size: u.mas, wl: u.um, **kwargs
    ) -> np.ndarray:
        """Computes the image from a 2D grid.

        Parameters
        ----------
        xx : u.mas
        yy : u.mas
        wavelength : u.um

        Returns
        -------
        image : astropy.units.Jy
        """
        if self.rin.unit == u.au:
            xx = (xx * 1e-3 * self.dist()).value * u.au
            yy = (yy * 1e-3 * self.dist()).value * u.au

        radius = np.hypot(xx, yy)[np.newaxis, ...]
        dx = np.max([np.diff(xx), np.diff(yy)]) * self.rin.unit
        if not self.thin:
            dx = self.rout() - self.rin()

        radial_profile = (radius >= self.rin()) & (radius <= (self.rin() + dx))
        intensity_func = kwargs.pop("intensity_func", None)
        if intensity_func is None:
            intensity = 1 / (2 * np.pi * dx.value)
        else:
            intensity = (
                intensity_func(radius, wl).to(u.erg / (u.cm**2 * u.rad**2 * u.s * u.Hz))
                * pixel_size.to(u.rad) ** 2
            )
            intensity = intensity.to(u.Jy)

        image = intensity * radial_profile
        if self.asymmetric:
            polar_angle, modulations = np.arctan2(yy, xx), []
            for i in range(1, OPTIONS.model.modulation + 1):
                rho, theta = getattr(self, f"rho{i}")(), getattr(self, f"theta{i}")()
                modulations.append(
                    rho
                    * np.cos(
                        compare_angles(theta.to(u.rad).value, i * polar_angle.value)
                    )
                )

            modulations = u.Quantity(modulations)
            image = image * (1 + np.sum(modulations, axis=0))

        return image.astype(OPTIONS.data.dtype.real)


class TempGrad(Ring):
    """The base class for the component.

    Parameters
    ----------
    xx : float
        The x-coordinate of the component.
    yy : float
        The x-coordinate of the component.
    dim : float
        The dimension [px].
    """

    name = "TempGrad"
    thin = False
    optically_thick = False
    const_temperature = False
    continuum_contribution = True

    def __init__(self, **kwargs):
        """The class's constructor."""
        super().__init__(**kwargs)
        self.rin.unit = self.rout.unit = u.au
        self.dist = Parameter(base="dist")
        self.eff_temp = Parameter(base="eff_temp")
        self.eff_radius = Parameter(base="eff_radius")

        self.r0 = Parameter(base="r0")
        self.q = Parameter(base="q")
        self.temp0 = Parameter(base="temp0")
        self.p = Parameter(base="p")
        self.sigma0 = Parameter(base="sigma0")

        self.weights, self.radii, self.matrix = None, None, None
        self.kappa_abs = Parameter(base="kappa_abs")
        self.kappa_cont = Parameter(base="kappa_cont")
        self.weight_cont = Parameter(base="weight_cont")

        if self.const_temperature:
            self.q.free = self.temp0.free = False

        if not self.continuum_contribution:
            self.weight_cont.free = False

        self.eval(**kwargs)

    def get_opacity(self, wl: u.um) -> u.cm**2 / u.g:
        """Set the opacity from wavelength."""
        kappa_abs = self.kappa_abs(wl)
        if self.continuum_contribution:
            cont_weight, kappa_cont = (
                self.weight_cont().value / 1e2,
                self.kappa_cont(wl),
            )
            opacity = (1 - cont_weight) * kappa_abs + cont_weight * kappa_cont
        else:
            opacity = kappa_abs

        opacity = opacity.astype(OPTIONS.data.dtype.real)
        if opacity.size == 1:
            return opacity.squeeze()

        return opacity

    def compute_temperature(self, radius: u.au) -> u.K:
        """Computes a 1D-temperature profile."""
        if self.const_temperature:
            if self.matrix is not None:
                interp_op_temps = interp1d(self.weights, self.matrix, axis=0)(
                    self.weight_cont().value / 1e2
                )
                temp = np.interp(radius.value, self.radii, interp_op_temps) * u.K
            else:
                temp = (
                    np.sqrt(self.eff_radius().to(u.au) / (2 * radius)) * self.eff_temp()
                )
        else:
            temp = self.temp0() * (radius / self.r0()) ** self.q()
        return temp.astype(OPTIONS.data.dtype.real)

    def compute_surface_density(self, radius: u.au) -> u.one:
        """Computes a 1D-surface density profile."""
        sigma = self.sigma0() * (radius / self.r0()) ** self.p()
        return sigma.astype(OPTIONS.data.dtype.real)

    def compute_optical_depth(self, radius: u.au, wl: u.um) -> u.one:
        """Computes a 1D-optical depth profile."""
        tau = self.compute_surface_density(radius) * self.get_opacity(wl)
        return tau.astype(OPTIONS.data.dtype.real)

    def compute_emissivity(self, radius: u.au, wl: u.um) -> u.one:
        """Computes a 1D-emissivity profile."""
        if wl.shape == ():
            wl.reshape((wl.size,))

        if self.optically_thick:
            return np.array([1])[:, np.newaxis]

        tau = self.compute_optical_depth(radius, wl)
        epsilon = 1 - np.exp(-tau / self.cinc())
        return epsilon.astype(OPTIONS.data.dtype.real)

    def compute_intensity(self, radius: u.au, wl: u.um) -> u.Jy:
        """Computes a 1D-brightness profile from a dust-surface density- and
        temperature profile.

        Parameters
        ----------
        wl : astropy.units.um
            Wavelengths.

        Returns
        -------
        brightness_profile : astropy.units.Jy
        """
        temperature = self.compute_temperature(radius)
        emissivity = self.compute_emissivity(radius, wl)
        intensity = BlackBody(temperature)(wl) * emissivity
        return intensity.astype(OPTIONS.data.dtype.real)

    def flux_func(self, wl: u.um) -> np.ndarray:
        """Computes the total flux from the hankel transformation."""
        radius = self.compute_internal_grid()
        intensity = self.compute_intensity(radius, wl[:, np.newaxis])
        if self.rin.unit == u.au:
            radius = (radius.to(u.au) / self.dist().to(u.pc)).value * 1e3 * u.mas

        flux = 2 * np.pi * self.cinc() * np.trapz(radius * intensity, radius).to(u.Jy)
        return flux.value.reshape((flux.shape[0], 1)).astype(OPTIONS.data.dtype.real)

    def vis_func(self, *args) -> np.ndarray:
        """Computes the correlated fluxes via the hankel transformation.

        Parameters
        ----------
        radius : astropy.units.mas
            The radius.
        baseline : 1/astropy.units.rad
            The deprojected baselines.
        baseline_angles : astropy.units.rad
            The deprojected baseline angles.
        wavelength : astropy.units.um
            The wavelengths.

        Returns
        -------
        vis : numpy.ndarray
            The correlated fluxes.
        """
        return super().vis_func(*args, intensity_func=self.compute_intensity)

    def image_func(self, *args) -> np.ndarray:
        """Computes the image."""
        return super().image_func(*args, intensity_func=self.compute_intensity)


class AsymTempGrad(TempGrad):
    """An analytical implementation of an asymmetric temperature
    gradient."""

    name = "AsymTempGrad"
    asymmetric = True


class GreyBody(TempGrad):
    """An analytical implementation of an asymmetric temperature
    gradient."""

    name = "GreyBody"
    const_temperature = True


class AsymGreyBody(GreyBody):
    """An analytical implementation of an asymmetric temperature
    gradient."""

    name = "AsymGreyBody"
    asymmetric = True
