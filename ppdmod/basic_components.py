import sys
from typing import Optional, Dict, List
import astropy.units as u
import numpy as np
from astropy.modeling.models import BlackBody
from scipy.special import j0, j1, jv
from scipy.signal import fftconvolve

from .component import Component
from .parameter import Parameter
from .options import STANDARD_PARAMETERS, OPTIONS
from .utils import distance_to_angular


class PointSource(Component):
    """A point source that can contain a relative flux contribution.

    Parameters
    ----------
    x : int
        x pos of the component (mas).
    y : int
        y pos of the component (mas).
    f : float
        Relative flux contribution (percent).

    Attributes
    ----------
    name : str
        The component's name.
    shortname : str
        The component's short name.
    description : str
        The component's description.
    """

    name = "Point Source"
    shortname = "Point"
    description = "Point source with flux contribution."
    elliptic = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.eval(**kwargs)

    def flux_func(self, wavelength: u.um) -> np.ndarray:
        """Returns the flux weight of the point source."""
        return self.fr(wavelength).value.reshape((wavelength.size, 1))

    def vis_func(self, baselines: 1 / u.rad, baseline_angles: u.rad,
                 wavelength: u.um, **kwargs) -> np.ndarray:
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

    # TODO: Change the way that is implemented, as it is always in the centre right now
    def image_func(self, xx: u.mas, yy: u.mas,
                   pixel_size: u.mas, wavelength: u.m = None) -> np.ndarray:
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
        val = np.abs(xx) + np.abs(yy)
        idx = np.unravel_index(np.argmin(val), np.shape(val))
        image[:, idx[0], idx[1]] = self.compute_flux(wavelength)
        return image


class Star(Component):
    """Star defined as component.

    Parameters
    ----------
    x : int
        x pos of the component (mas).
    y : int
        y pos of the component (mas).
    dist : float
        Distance to the star.
    eff_temp : float
        The star's temperature.
    eff_radius : float
        The star's radius.

    Attributes
    ----------
    name : str
        The component's name.
    shortname : str
        The component's short name.
    description : str
        The component's description.
     : dict of Parameter
    stellar_radius_angular : u.mas
    """

    name = "Star"
    shortname = "Star"
    description = "The flux of a star."
    elliptic = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._stellar_angular_radius = None

        self.f = Parameter(**STANDARD_PARAMETERS.f)
        self.dist = Parameter(**STANDARD_PARAMETERS.dist)
        self.eff_temp = Parameter(**STANDARD_PARAMETERS.eff_temp)
        self.eff_radius = Parameter(**STANDARD_PARAMETERS.eff_radius)
        self.eval(**kwargs)

    @property
    def stellar_radius_angular(self) -> u.mas:
        r"""Computes the parallax from the stellar radius and the distance to
        the object.

        Returns
        -------
        stellar_radius_angular : astropy.units.mas
            The parallax of the stellar radius.
        """
        self._stellar_angular_radius = distance_to_angular(
            self.eff_radius(), self.dist())
        return self._stellar_angular_radius

    def flux_func(self, wavelength: u.um) -> np.ndarray:
        """Computes the flux of the star."""
        if self.f.value is not None:
            stellar_flux = self.f(wavelength)
        else:
            plancks_law = BlackBody(temperature=self.eff_temp())
            spectral_radiance = plancks_law(wavelength).to(
                u.erg / (u.cm**2 * u.Hz * u.s * u.rad**2))
            stellar_flux = np.pi * (
                spectral_radiance * self.stellar_radius_angular**2
            ).to(u.Jy)
        return stellar_flux.value.reshape((wavelength.size, 1))

    def vis_func(self, baselines: 1 / u.rad, baseline_angles: u.rad,
                 wavelength: u.um, **kwargs) -> np.ndarray:
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
            self.flux_func(wavelength).reshape(new_shape), baselines.shape[1:])
        return vis.astype(OPTIONS.data.dtype.complex)

    def image_func(self, xx: u.mas, yy: u.mas,
                   pixel_size: u.mas, wavelength: u.m = None) -> np.ndarray:
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


class Ring(Component):
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
    shortname = "Ring"
    description = "A simple ring."
    thin, has_outer_radius = True, False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rin = Parameter(**STANDARD_PARAMETERS.rin)
        self.rout = Parameter(**STANDARD_PARAMETERS.rout)
        self.width = Parameter(**STANDARD_PARAMETERS.width)

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
        return radius.astype(OPTIONS.data.dtype.real) * u.mas

    def vis_func(self, baselines: 1 / u.rad, baseline_angles: u.rad,
                 wavelength: u.um, **kwargs) -> np.ndarray:
        """Computes the complex visibility

        Parameters
        ----------
        baseline : 1/astropy.units.rad
            The deprojected baselines.
        baseline_angles : astropy.units.rad
            The deprojected baseline angles.
        wavelength : astropy.units.um
            The wavelengths.
        brightness : astropy.unit.mas
            The radial brightness distribution
        """
        mod_amp = np.hypot(self.c1(), self.s1())
        mod_angle = np.arctan2(self.c1(), self.s1()) - 180 * u.deg
        angle_diff = np.angle(np.exp(1j*(baseline_angles - mod_angle).value))

        # TODO: Check if this is too much overhead
        def _vis_func(xx: np.ndarray):
            """Shorthand for the vis calculation."""
            vis = j0(xx).astype(complex)
            if self.asymmetric:
                vis += -1j * mod_amp * np.cos(angle_diff) * j1(xx)
            return vis

        if self.thin:
            vis = _vis_func(2 * np.pi * self.rin().to(u.rad) * baselines)
        else:
            intensity, intensity_func = 1, kwargs.pop("intensity_func", None)
            radius = self.compute_internal_grid()
            if intensity_func is not None:
                intensity = intensity_func(radius, wavelength).to(u.erg/(u.rad**2 * u.cm**2 * u.s * u.Hz))
                intensity = intensity[:, np.newaxis]

            radius = radius.to(u.rad)
            vis = _vis_func(2 * np.pi * radius * baselines)
            vis = np.trapz(radius * intensity * vis, radius).to(u.Jy)

            # TODO: Maybe move the factor of 2pi to the visibility where it belongs?
            if intensity_func is None:
                if self.has_outer_radius:
                    vis /= (self.rout() - self.rin()).to(u.rad)
                else:
                    vis /= self.width().to(u.rad)
            else:
                vis *= 2 * np.pi * self.inc()

            vis = vis[..., np.newaxis]
        return vis.value.astype(OPTIONS.data.dtype.complex)

    def image_func(self, xx: u.mas, yy: u.mas,
                   pixel_size: u.mas, wavelength: u.um,
                   **kwargs) -> np.ndarray:
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
        radius = np.hypot(xx, yy)[np.newaxis, ...]

        dx = np.max([np.diff(xx), np.diff(yy)])*u.mas
        if not self.thin:
            dx = self.rout() - self.rin() if self.has_outer_radius else self.width()

        radial_profile = (radius >= self.rin()) & (radius <= (self.rin() + dx))
        intensity_func = kwargs.pop("intensity_func", None)
        if intensity_func is None:
            intensity = 1 / (2 * np.pi)
        else:
            intensity = intensity_func(radius, wavelength).to(
                u.erg / (u.cm**2 * u.rad**2 * u.s * u.Hz)) * pixel_size.to(u.rad) ** 2
            intensity = intensity.to(u.Jy)

        image = intensity * radial_profile
        if self.asymmetric:
            polar_angle = np.arctan2(xx, yy)
            image *= 1 + self.c1() * np.cos(polar_angle) + self.s1() * np.sin(polar_angle)

        return image.astype(OPTIONS.data.dtype.real)


class UniformDisk(Component):
    name = "Uniform Disk"
    shortname = "UniformDisk"
    description = "A uniform disk."

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.diam = Parameter(**STANDARD_PARAMETERS.diam)
        self.eval(**kwargs)

    def vis_func(self, baselines: 1 / u.rad, baseline_angles: u.rad,
                 wavelength: u.um, **kwargs) -> np.ndarray:
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
        vis = (
            2
            * j1(np.pi * self.diam().to(u.rad) * baselines)
            / (np.pi * self.diam().to(u.rad) * baselines)
        )
        return vis.value.astype(OPTIONS.data.dtype.complex)

    def image_func(
        self, xx: u.mas, yy: u.mas, pixel_size: u.mas, wavelength: u.um
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
        radius = np.hypot(xx, yy)[np.newaxis, ...]
        return (radius <= self.diam() / 2).astype(OPTIONS.data.dtype.real)


class Gaussian(Component):
    name = "Gaussian"
    shortname = "Gaussian"
    description = "A simple 2D Gaussian."

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hlr = Parameter(**STANDARD_PARAMETERS.hlr)
        self.eval(**kwargs)

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
        xx = np.pi * self.hlr().to(u.rad) * baselines
        vis = np.exp(-xx**2 / np.log(2))
        return vis.value.astype(OPTIONS.data.dtype.complex)

    def image_func(self, xx: u.mas, yy: u.mas,
                   pixel_size: u.mas, wavelength: u.um) -> np.ndarray:
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
        hlr = 2 * self.hlr() * (1/self.inc())
        radius = np.hypot(xx, yy)[np.newaxis, ...]
        image = np.log(2) / (np.pi * hlr ** 2) \
            * np.exp(-(radius / hlr) ** 2 * np.log(2))
        return image.value.astype(OPTIONS.data.dtype.real)


class Lorentzian(Component):
    name = "Lorentzian"
    shortname = "Lorentzian"
    description = "A simple 2D Lorentzian."

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hlr = Parameter(**STANDARD_PARAMETERS.hlr)
        self.eval(**kwargs)

    def vis_func(self, baselines: 1 / u.rad, baseline_angles: u.rad,
                 wavelength: u.um, **kwargs) -> np.ndarray:
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
        xx = np.pi * self.hlr().to(u.rad) * baselines
        vis = np.exp(-2 * xx / np.sqrt(3))
        return vis.value.astype(OPTIONS.data.dtype.complex)

    def image_func(self, xx: u.mas, yy: u.mas,
                   pixel_size: u.mas, wavelength: u.um) -> np.ndarray:
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
        radius = np.hypot(xx, yy)[np.newaxis, ...]
        hlr = self.hlr() * self.inc()
        image = hlr / (2 * np.pi**2 * np.sqrt(3)) \
            * (hlr ** 2 / 3 + radius**2) ** (-3 / 2)
        return image.value.astype(OPTIONS.data.dtype.real)


class GaussLorentzian(Component):
    name = "Gauss-Lorentzian"
    shortname = "GaussLorentzian"
    description = "A simple 2D Gaussian combined with a Lorentzian."

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.flor = Parameter(**STANDARD_PARAMETERS.fr)
        self.flor.name = self.flor.shortname = "flor"
        self.flor.free = True
        self.hlr = Parameter(**STANDARD_PARAMETERS.hlr)

        self.eval(**kwargs)

        self.gauss = Gaussian(hlr=self.hlr, inc=self.inc, pa=self.pa)
        self.lor = Lorentzian(hlr=self.hlr, inc=self.inc, pa=self.pa)

    def vis_func(self, baselines: 1 / u.rad, baseline_angles: u.rad,
                 wavelength: u.um, **kwargs) -> np.ndarray:
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
        vis_gauss = self.gauss.vis_func(
            baselines, baseline_angles, wavelength, **kwargs)
        vis_lor = self.lor.vis_func(baselines, baseline_angles, wavelength, **kwargs)
        vis = (1 - self.flor()) * vis_gauss + self.flor() * vis_lor
        return vis.value.astype(OPTIONS.data.dtype.complex)

    def image_func(self, xx: u.mas, yy: u.mas,
                   pixel_size: u.mas, wavelength: u.um) -> np.ndarray:
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
        image_gauss = self.gauss.image_func(xx, yy, pixel_size, wavelength)
        image_lor = self.lor.image_func(xx, yy, pixel_size, wavelength)
        image = (1 - self.flor()) * image_gauss + self.flor() * image_lor
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
    name = "Temperature Gradient"
    shortname = "TempGrad"
    asymmetric = False
    thin = False
    has_outer_radius = True
    optically_thick = False
    const_temperature = False
    continuum_contribution = True

    def __init__(self, **kwargs):
        """The class's constructor."""
        super().__init__(**kwargs)

        self.dist = Parameter(**STANDARD_PARAMETERS.dist)
        self.eff_temp = Parameter(**STANDARD_PARAMETERS.eff_temp)
        self.eff_radius = Parameter(**STANDARD_PARAMETERS.eff_radius)

        self.r0 = Parameter(**STANDARD_PARAMETERS.r0)
        self.q = Parameter(**STANDARD_PARAMETERS.q)
        self.inner_temp = Parameter(**STANDARD_PARAMETERS.inner_temp)

        self.p = Parameter(**STANDARD_PARAMETERS.p)
        self.inner_sigma = Parameter(**STANDARD_PARAMETERS.inner_sigma)
        self.kappa_abs = Parameter(**STANDARD_PARAMETERS.kappa_abs)

        self.cont_weight = Parameter(**STANDARD_PARAMETERS.cont_weight)
        self.kappa_cont = Parameter(**STANDARD_PARAMETERS.kappa_cont)

        if self.const_temperature:
            self.q.free = False
            self.inner_temp.free = False

        if not self.continuum_contribution:
            self.cont_weight.free = False

        self.eval(**kwargs)

    @property
    def stellar_radius_angular(self) -> u.mas:
        r"""Computes the parallax from the stellar radius and the distance to
        the object.

        Returns
        -------
        stellar_radius_angular : astropy.units.mas
            The parallax of the stellar radius.
        """
        return distance_to_angular(self.eff_radius(), self.dist())

    def get_opacity(self, wavelength: u.um) -> u.cm**2 / u.g:
        """Set the opacity from wavelength."""
        kappa_abs = self.kappa_abs(wavelength)
        if self.continuum_contribution:
            cont_weight, kappa_cont = self.cont_weight(), self.kappa_cont(wavelength)
            opacity = (1 - cont_weight) * kappa_abs + cont_weight * kappa_cont
        else:
            opacity = kappa_abs

        opacity = opacity.astype(OPTIONS.data.dtype.real)
        if opacity.size == 1:
            return opacity.squeeze()

        shape = tuple(np.newaxis for _ in range(len(wavelength.shape) - 1))
        return opacity[(slice(None), *shape)]

    def compute_temperature(self, radius: u.mas) -> u.K:
        """Computes a 1D-temperature profile."""
        if self.r0.value != 0:
            reference_radius = distance_to_angular(self.r0(), self.dist())
        else:
            reference_radius = distance_to_angular(
                OPTIONS.model.reference_radius, self.dist()
            )

        if self.const_temperature:
            temperature = (
                np.sqrt(self.stellar_radius_angular / (2.0 * radius)) * self.eff_temp()
            )
        else:
            temperature = self.inner_temp() * (radius / reference_radius) ** -self.q()
        return temperature.astype(OPTIONS.data.dtype.real)

    def compute_surface_density(self, radius: u.mas) -> u.one:
        """Computes a 1D-surface density profile."""
        if self.r0.value != 0:
            reference_radius = distance_to_angular(self.r0(), self.dist())
        else:
            reference_radius = distance_to_angular(
                OPTIONS.model.reference_radius, self.dist()
            )

        surface_density = self.inner_sigma() * (radius / reference_radius) ** -self.p()
        return surface_density.astype(OPTIONS.data.dtype.real)

    def compute_emissivity(self, radius: u.mas, wavelength: u.um) -> u.one:
        """Computes a 1D-emissivity profile."""
        if wavelength.shape == ():
            wavelength.reshape((wavelength.size,))

        if self.optically_thick:
            return np.array([1])[:, np.newaxis]

        surface_density = self.compute_surface_density(radius)
        optical_depth = surface_density * self.get_opacity(wavelength)
        emissivity = 1 - np.exp(-optical_depth / self.inc())
        return emissivity.astype(OPTIONS.data.dtype.real)

    def compute_intensity(self, radius: u.mas, wavelength: u.um) -> u.Jy:
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
        brightness = BlackBody(temperature)(wavelength)
        emissivity = self.compute_emissivity(radius, wavelength)
        return (brightness * emissivity).astype(OPTIONS.data.dtype.real)

    def flux_func(self, wavelength: u.um) -> np.ndarray:
        """Computes the total flux from the hankel transformation."""
        radius = self.compute_internal_grid()
        intensity = self.compute_intensity(radius, wavelength[:, np.newaxis])
        flux = 2 * np.pi * self.inc() * np.trapz(radius * intensity, radius).to(u.Jy)
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
        # TODO: Check if a factor of 2*np.pi is required in front of the integration
        return super().vis_func(*args, intensity_func=self.compute_intensity)

    def image_func(self, *args) -> np.ndarray:
        """Computes the image."""
        return super().image_func(*args, intensity_func=self.compute_intensity)


class AsymmetricTempGradient(TempGradient):
    """An analytical implementation of an asymmetric temperature
    gradient."""

    name = "Asymmetric Continuum Grey Body"
    shortname = "AsymContinuumGreyBody"
    asymmetric = True


class GreyBody(TempGradient):
    """An analytical implementation of an asymmetric temperature
    gradient."""

    name = "Symmetric Continuum Grey Body"
    shortname = "ContinuumGreyBody"
    const_temperature = True


class AsymmetricGreyBody(GreyBody):
    """An analytical implementation of an asymmetric temperature
    gradient."""

    name = "Asymmetric Continuum Grey Body"
    shortname = "AsymContinuumGreyBody"
    asymmetric = True

class StarHaloGauss(Component):
    """A star, a disk and a halo model with a Gauss profile.

    From Lazareff+2017.
    """

    name = "StarHaloGauss"
    shortname = "StarHaloGauss"
    is_gauss_lor = False
    has_ring = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fs = Parameter(**STANDARD_PARAMETERS.fr)
        self.fs.name = self.fs.shortname = "fs"
        self.fc = Parameter(**STANDARD_PARAMETERS.fr)
        self.fc.name = self.fc.shortname = "fc"
        self.fs.free = self.fc.free = True

        self.la = Parameter(**STANDARD_PARAMETERS.la)

        if self.has_ring:
            self.lkr = Parameter(**STANDARD_PARAMETERS.lkr)

        self.wl0 = Parameter(**STANDARD_PARAMETERS.wl)
        self.ks = Parameter(**STANDARD_PARAMETERS.exp)
        self.ks.name = self.ks.shortname = "ks"
        self.ks.value = 1
        self.ks.min, self.ks.max = -10, 10
        self.ks.free = False
        self.kc = Parameter(**STANDARD_PARAMETERS.exp)
        self.kc.name = self.kc.shortname = "kc"
        self.kc.min, self.kc.max = -10, 10
        self.kc.value = 1

        if self.is_gauss_lor:
            self.flor = Parameter(**STANDARD_PARAMETERS.fr)
            self.flor.name = self.flor.shortname = "flor"
            self.flor.free = True

        self.eval(**kwargs)
        self.hlr = 10 ** self.la()
        if self.has_ring:
            self.rin = np.sqrt(10 ** (2 * self.la()) / (1 + 10 ** (2 * self.lkr())))
            self.hlr = np.sqrt(10 ** (2 * self.la()) / (1 + 10 ** (-2 * self.lkr())))

        if self.is_gauss_lor:
            self.comp = GaussLorentzian(
                flor=self.flor, hlr=self.hlr, inc=self.inc, pa=self.pa)
        else:
            self.comp = Gaussian(hlr=self.hlr, inc=self.inc, pa=self.pa)

        if self.has_ring:
            self.ring = Ring(
                rin=self.rin, inc=self.inc, pa=self.pa,
                asymmetric=True, c1=self.c1, s1=self.s1)

    def vis_func(self, baselines: 1 / u.rad, baseline_angles: u.rad,
                 wavelength: u.um, **kwargs) -> np.ndarray:
        if self.wl0() == 0:
            self.wl0.value = np.mean(wavelength)

        wavelength_ratio = self.wl0() / wavelength
        fs, ks = self.fs(wavelength), self.ks(wavelength)
        complex_vis_star = (fs * wavelength_ratio**ks).astype(OPTIONS.data.dtype.complex)
        component_ratio = (1-self.fc()) * wavelength_ratio**ks \
            + self.fc() * wavelength_ratio**self.kc()

        complex_vis_comp = self.comp.vis_func(
            baselines, baseline_angles, wavelength, **kwargs)

        if self.has_ring:
            complex_vis_ring = self.ring.vis_func(
                baselines, baseline_angles, wavelength, **kwargs)
            complex_vis_comp *= complex_vis_ring

        complex_vis_comp = self.fc() * complex_vis_comp \
            * wavelength_ratio[..., np.newaxis]**self.kc()
        vis = (complex_vis_star[..., np.newaxis] + complex_vis_comp) \
            / component_ratio[..., np.newaxis]
        return vis.value.astype(OPTIONS.data.dtype.complex)

    def image_func(self, xx: u.mas, yy: u.mas,
                   pixel_size: u.mas, wavelength: u.um) -> np.ndarray:
        """Computes the image."""
        fh = 1 - (self.fs(wavelength) + self.fc())
        image = self.comp.image_func(xx, yy, pixel_size, wavelength)
        if self.has_ring:
            image_ring = self.ring.image_func(xx, yy, pixel_size, wavelength)
            image = self.fc() * fftconvolve(image, image_ring, mode="same")

        pt = PointSource(inc=self.inc, pa=self.pa)
        image += self.fs(wavelength) * pt.image_func(xx, yy, pixel_size, wavelength) + fh
        image /= total if (total := np.sum(image, axis=(-2, -1))) != 0 else 1
        return (self.fr() * image).astype(OPTIONS.data.dtype.real)

class StarHaloGaussLor(StarHaloGauss):
    """A star, a disk and a halo model with a Gauss-Lorentzian profile.

    From Lazareff+2017.
    """

    name = "StarHaloGaussLor"
    shortname = "StarHaloGaussLor"
    is_gauss_lor = True
    has_ring = False


class StarHaloRing(StarHaloGaussLor):
    """A star, a disk and a halo model with a Gauss-Lorentzian profile convolved with a ring.

    From Lazareff+2017.
    """

    name = "StarHaloRing"
    shortname = "StarHaloRing"
    has_ring = True


def assemble_components(parameters: Dict[str, Dict],
                        shared_params: Optional[Dict[str, Parameter]] = None
                        ) -> List[Component]:
    """Assembles a model from a dictionary of parameters."""
    shared_params = shared_params if shared_params is not None else {}
    if OPTIONS.model.constant_params is None:
        constant_params = {}
    else:
        constant_params = OPTIONS.model.constant_params

    components = []
    for component, params in parameters:
        comp = getattr(sys.modules[__name__], component)
        components.append(comp(**params, **shared_params, **constant_params))
    return components
