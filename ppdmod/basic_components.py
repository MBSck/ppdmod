from functools import partial

import astropy.units as u
import numpy as np
from astropy.modeling.models import BlackBody
from scipy.interpolate import interp1d
from scipy.special import j0, jv

from .component import Component, FourierComponent
from .options import OPTIONS
from .parameter import Parameter
from .utils import angular_to_distance, distance_to_angular, compare_angles


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


class Star(FourierComponent):
    """Star defined as component.

    Parameters
    ----------
    f : astropy.units.Jy
        The flux of the star.
    dist : astropy.units.pc
        Distance to the star.
    eff_temp : astropy.units.K
        The star's temperature.
    eff_radius : astropy.units.R_sun
        The star's radius.

    Attributes
    ----------
    name : str
        The component's name.
    description : str
        The component's description.
     : dict of Parameter
    """

    name = "Star"
    description = "The flux of a star."

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.f = Parameter(base="f")
        self.dist = Parameter(base="dist")
        self.eff_temp = Parameter(base="eff_temp")
        self.eff_radius = Parameter(base="eff_radius")
        self.eval(**kwargs)

        self.stellar_radius_angular = distance_to_angular(
            self.eff_radius(), self.dist()
        )

    def flux_func(self, wavelength: u.um) -> np.ndarray:
        """Computes the flux of the star."""
        if np.any(self.f.value != 0):
            stellar_flux = self.f(wavelength)
        else:
            plancks_law = BlackBody(temperature=self.eff_temp())
            spectral_radiance = plancks_law(wavelength).to(
                u.erg / (u.cm**2 * u.Hz * u.s * u.rad**2)
            )
            stellar_flux = np.pi * (
                spectral_radiance * self.stellar_radius_angular**2
            ).to(u.Jy)
        return stellar_flux.value.reshape((wavelength.size, 1))

    def vis_func(
        self, baselines: 1 / u.rad, baseline_angles: u.rad, wavelength: u.um, **kwargs
    ) -> np.ndarray:
        """Computes the complex visibility

        Parameters
        ----------
        baseline : 1/astropy.units.rad
            The deprojected baselines.
        baseline_angles : astropy.units.rad
            The deprojected baseline angles.
        wavelength : astropy.units.um
            The wavelengths.
        """
        new_shape = (-1,) + (1,) * (len(baselines.shape) - 1)
        vis = np.tile(
            self.flux_func(wavelength).reshape(new_shape), baselines.shape[1:]
        )
        return vis.astype(OPTIONS.data.dtype.complex)

    def image_func(
        self, xx: u.mas, yy: u.mas, pixel_size: u.mas, wavelength: u.m = None
    ) -> np.ndarray:
        """Computes the image from a 2D grid.

        Parameters
        ----------
        xx : u.mas
        yy : u.mas
        wavelength : u.m

        Returns
        -------
        image : astropy.units.Jy
        """
        image = np.zeros((wavelength.size, *xx.shape))
        centre = xx.shape[0] // 2
        star_flux = (self.compute_flux(wavelength) / 4)[..., np.newaxis]
        image[:, centre - 1 : centre + 1, centre - 1 : centre + 1] = star_flux
        return image


class Ring(FourierComponent):
    """A ring.

    Parameters
    ----------
    rin : astropy.units.mas
        The inner radius of the ring
    thin : bool
        If toggled the ring has an infinitesimal width.
        Default is 'True'.
    has_outer_radius : bool
        If toggled the ring has an outer radius instead of
        a width.
        Default is 'False'.
    width : astropy.units.mas
        The width of the ring. Applies only for 'False' thin
        and 'False' has outer radius.
    rout : astropy.units.mas
        The outer radius of the ring. Applies only for 'False' thin
    """

    name = "Ring"
    description = "A simple ring."
    thin, has_outer_radius = True, False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rin = Parameter(base="rin")
        self.rout = Parameter(base="rout")
        self.width = Parameter(base="width")

        if self.has_outer_radius or self.thin:
            self.width.free = False

        self.eval(**kwargs)

    def compute_internal_grid(self) -> u.mas:
        """Computes the model grid.

        Parameters
        ----------
        dim : float

        Returns
        -------
        radial_grid : astropy.units.mas
            A one dimensional linear or logarithmic grid.
        """
        dx = self.rout() - self.rin() if self.has_outer_radius else self.width()
        rin, rout = self.rin().value, (self.rin() + dx).value
        if OPTIONS.model.gridtype == "linear":
            radius = np.linspace(rin, rout, self.dim())
        else:
            radius = np.logspace(np.log10(rin), np.log10(rout), self.dim())
        return radius.astype(OPTIONS.data.dtype.real) * self.rin.unit

    def vis_func(
        self, baselines: 1 / u.rad, baseline_angles: u.rad, wavelength: u.um, **kwargs
    ) -> np.ndarray:
        """Computes the complex visibility

        Parameters
        ----------
        baseline : 1 / astropy.units.rad
            The deprojected baselines.
        baseline_angles : astropy.units.rad
            The deprojected baseline angles.
        wavelength : astropy.units.um
            The wavelengths.
        brightness : astropy.unit.mas
            The radial brightness distribution
        """
        mod_amps, cos_diff, bessel_funcs = [], [], []
        if self.asymmetric:
            for i in range(1, OPTIONS.model.modulation + 1):
                rho, theta = getattr(self, f"rho{i}")(), getattr(self, f"theta{i}")()
                mod_amps.append((-1j) ** i * rho)
                cos_diff.append(np.cos(i * compare_angles(baseline_angles, theta)))
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
            vis = _vis_func(2 * np.pi * self.rin().to(u.rad) * baselines)
        else:
            intensity_func = kwargs.pop("intensity_func", None)
            radius = self.compute_internal_grid()

            if intensity_func is not None:
                intensity = intensity_func(radius, wavelength).to(
                    u.erg / (u.rad**2 * u.cm**2 * u.s * u.Hz)
                )
                intensity = intensity[:, np.newaxis]

            if radius.unit not in [u.rad, u.mas]:
                radius = distance_to_angular(radius, self.dist())

            radius = radius.to(u.rad)
            vis = _vis_func(2 * np.pi * radius * baselines)
            if intensity_func is None:
                vis = np.trapezoid(vis, radius)
            else:
                vis = np.trapezoid(radius * intensity * vis, radius).to(u.Jy)

            if intensity_func is None:
                if self.has_outer_radius:
                    vis /= (self.rout() - self.rin()).to(u.rad)
                else:
                    vis /= self.width().to(u.rad)
            else:
                vis *= 2 * np.pi * self.cinc()

        return vis.value.astype(OPTIONS.data.dtype.complex)

    def image_func(
        self, xx: u.mas, yy: u.mas, pixel_size: u.mas, wavelength: u.um, **kwargs
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
            xx = angular_to_distance(xx, self.dist()).to(u.au)
            yy = angular_to_distance(yy, self.dist()).to(u.au)

        radius = np.hypot(xx, yy)[np.newaxis, ...]
        dx = np.max([np.diff(xx), np.diff(yy)]) * self.rin.unit
        if not self.thin:
            dx = self.rout() - self.rin() if self.has_outer_radius else self.width()

        radial_profile = (radius >= self.rin()) & (radius <= (self.rin() + dx))
        intensity_func = kwargs.pop("intensity_func", None)
        if intensity_func is None:
            intensity = 1 / (2 * np.pi)
        else:
            intensity = (
                intensity_func(radius, wavelength).to(
                    u.erg / (u.cm**2 * u.rad**2 * u.s * u.Hz)
                )
                * pixel_size.to(u.rad) ** 2
            )
            intensity = intensity.to(u.Jy)

        image = intensity * radial_profile
        if self.asymmetric:
            polar_angle, modulations = np.arctan2(yy, xx), []
            for i in range(1, OPTIONS.model.modulation + 1):
                rho, theta = getattr(self, f"rho{i}")(), getattr(self, f"theta{i}")()
                modulations.append(rho * np.cos(compare_angles(theta, i * polar_angle)))

            modulations = u.Quantity(modulations)
            image = image * (1 + np.sum(modulations, axis=0))

        return image.astype(OPTIONS.data.dtype.real)


class TempGradient(Ring):
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
    has_outer_radius = True
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

    def get_opacity(self, wavelength: u.um) -> u.cm**2 / u.g:
        """Set the opacity from wavelength."""
        kappa_abs = self.kappa_abs(wavelength)
        if self.continuum_contribution:
            cont_weight, kappa_cont = (
                self.weight_cont().value / 1e2,
                self.kappa_cont(wavelength),
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
                temperature = np.interp(radius.value, self.radii, interp_op_temps) * u.K
            else:
                temperature = (
                    np.sqrt(self.eff_radius().to(u.au) / (2 * radius)) * self.eff_temp()
                )
        else:
            temperature = self.temp0() * (radius / self.r0()) ** self.q()
        return temperature.astype(OPTIONS.data.dtype.real)

    def compute_surface_density(self, radius: u.au) -> u.one:
        """Computes a 1D-surface density profile."""
        surface_density = self.sigma0() * (radius / self.r0()) ** self.p()
        return surface_density.astype(OPTIONS.data.dtype.real)

    def compute_optical_depth(self, radius: u.au, wavelength: u.um) -> u.one:
        """Computes a 1D-optical depth profile."""
        surface_density = self.compute_surface_density(radius)
        optical_depth = surface_density * self.get_opacity(wavelength)
        return optical_depth.astype(OPTIONS.data.dtype.real)

    def compute_emissivity(self, radius: u.au, wavelength: u.um) -> u.one:
        """Computes a 1D-emissivity profile."""
        if wavelength.shape == ():
            wavelength.reshape((wavelength.size,))

        if self.optically_thick:
            return np.array([1])[:, np.newaxis]

        optical_depth = self.compute_optical_depth(radius, wavelength)
        emissivity = 1 - np.exp(-optical_depth / self.cinc())
        return emissivity.astype(OPTIONS.data.dtype.real)

    def compute_intensity(self, radius: u.au, wavelength: u.um) -> u.Jy:
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
        emissivity = self.compute_emissivity(radius, wavelength)
        intensity = BlackBody(temperature)(wavelength) * emissivity
        return intensity.astype(OPTIONS.data.dtype.real)

    def flux_func(self, wavelength: u.um) -> np.ndarray:
        """Computes the total flux from the hankel transformation."""
        radius = self.compute_internal_grid()
        intensity = self.compute_intensity(radius, wavelength[:, np.newaxis])
        if self.rin.unit == u.au:
            radius = distance_to_angular(radius, self.dist())

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


class AsymTempGradient(TempGradient):
    """An analytical implementation of an asymmetric temperature
    gradient."""

    name = "AsymTempGrad"
    asymmetric = True


class GreyBody(TempGradient):
    """An analytical implementation of an asymmetric temperature
    gradient."""

    name = "GreyBody"
    const_temperature = True


class AsymGreyBody(GreyBody):
    """An analytical implementation of an asymmetric temperature
    gradient."""

    name = "AsymGreyBody"
    asymmetric = True
