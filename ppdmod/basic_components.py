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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.elliptic = False
        self.eval(**kwargs)

    def flux_func(self, wavelength: u.um) -> np.ndarray:
        """Returns the flux weight of the point source."""
        return self.fr(wavelength).value.reshape((wavelength.size, 1))

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

    # TODO: Change the way that is implemented, as it is always in the centre right now
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._stellar_angular_radius = None
        self.elliptic = False

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
            self.eff_radius(), self.dist()
        )
        return self._stellar_angular_radius

    def flux_func(self, wavelength: u.um) -> np.ndarray:
        """Computes the flux of the star."""
        if self.f.value is not None:
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
    thin = True
    has_outer_radius = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rin = Parameter(**STANDARD_PARAMETERS.rin)

        if not self.thin:
            if self.has_outer_radius:
                self.rout = Parameter(**STANDARD_PARAMETERS.rout)
            else:
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

    def vis_func(
        self, baselines: 1 / u.rad,
        baseline_angles: u.rad, wavelength: u.um, **kwargs) -> np.ndarray:
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
        # brightness = kwargs.pop("brightness", 1)
        angle_diff = baseline_angles - (self.phi()-90*u.deg).to(u.rad)
        if self.thin:
            xx = 2 * np.pi * self.rin().to(u.rad) * baselines
            vis = j0(xx).astype(complex)
            if self.asymmetric:
                vis += -1j * self.a() * np.cos(angle_diff) * j1(xx)
        else:
            # TODO: Finish this with the factor here and use it for the temperature gradient
            # TODO: Check the integration and brightness, is it proper to integrate over the
            # u.rad or u.mas?
            radius = self.compute_internal_grid().to(u.rad)
            xx = 2 * np.pi * radius.to(u.rad) * baselines

            vis = (np.trapz(j0(xx), radius)).astype(complex)
            if self.asymmetric:
                vis += (-1j * self.a() * np.cos(angle_diff)).reshape(1, *vis.shape[1:]) \
                    * np.trapz(j1(xx), radius)

            # if brightness != 1:
            #     if not self.has_outer_radius:
            #         vis /= self.width()
            #     else:
            #         vis /= self.rout() - self.rin()

            vis = vis[..., np.newaxis]
        return vis.value.astype(OPTIONS.data.dtype.complex)

    def image_func(
        self, xx: u.mas, yy: u.mas,
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
        radius, dx = np.hypot(xx, yy)[np.newaxis, ...], pixel_size
        if not self.thin:
            dx = self.rout() - self.rin() if self.has_outer_radius else self.width()

        radial_profile = np.logical_and(radius >= self.rin(), radius <= (self.rin() + dx))
        image = (1 / (2 * np.pi)) * radius * radial_profile

        if self.asymmetric:
            phi = (self.phi-90*u.deg).to(u.rad)
            c, s = self.a() * np.cos(phi), self.a() * np.sin(phi)
            polar_angle = np.arctan2(yy, xx)
            image *= 1 + c * np.cos(polar_angle) + s * np.sin(polar_angle)

        return image.astype(OPTIONS.data.dtype.real)


class UniformDisk(Component):
    name = "Uniform Disk"
    shortname = "UniformDisk"
    description = "A uniform disk."

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.diam = Parameter(**STANDARD_PARAMETERS.diam)
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
        xx = np.pi * baselines * self.hlr().to(u.rad)
        vis = np.exp(-(xx**2) / np.log(2))
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
        image = (
            np.log(2)
            / (np.pi * self.hlr() ** 2)
            * np.exp(-((radius / self.hlr()) ** 2) * np.log(2))
        )
        return image.value.astype(OPTIONS.data.dtype.real)


class Lorentzian(Component):
    name = "Lorentzian"
    shortname = "Lorentzian"
    description = "A simple 2D Lorentzian."

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
        xx = np.pi * baselines * self.hlr().to(u.rad)
        vis = np.exp(-2 * xx / np.sqrt(3))
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
        image = (
            self.hlr()
            / (2 * np.pi * np.sqrt(3))
            * (self.hlr() ** 2 / 3 + radius**2) ** (-3 / 2)
        )
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
        vis_gauss = self.gauss.vis_func(
            baselines, baseline_angles, wavelength, **kwargs
        )
        vis_lor = self.lor.vis_func(baselines, baseline_angles, wavelength, **kwargs)
        vis = (1 - self.flor()) * vis_gauss + self.flor() * vis_lor
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
        image_gauss = self.gauss.image_func(xx, yy, pixel_size, wavelength)
        image_lor = self.lor.image_func(xx, yy, pixel_size, wavelength)
        image = (1 - self.flor()) * image_gauss + self.flor() * image_lor
        return image.astype(OPTIONS.data.dtype.real)


# TODO: Use the Ring as a base for the temperature gradient
class TempGradient(Component):
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
    elliptic = True
    optically_thick = False
    const_temperature = False
    continuum_contribution = True

    def __init__(self, **kwargs):
        """The class's constructor."""
        super().__init__(**kwargs)
        self._stellar_angular_radius = None

        self.dist = Parameter(**STANDARD_PARAMETERS.dist)
        self.eff_temp = Parameter(**STANDARD_PARAMETERS.eff_temp)
        self.eff_radius = Parameter(**STANDARD_PARAMETERS.eff_radius)

        self.r0 = Parameter(**STANDARD_PARAMETERS.r0)
        self.rin = Parameter(**STANDARD_PARAMETERS.rin)
        self.rout = Parameter(**STANDARD_PARAMETERS.rout)

        self.q = Parameter(**STANDARD_PARAMETERS.q)
        self.inner_temp = Parameter(**STANDARD_PARAMETERS.inner_temp)

        self.p = Parameter(**STANDARD_PARAMETERS.p)
        self.inner_sigma = Parameter(**STANDARD_PARAMETERS.inner_sigma)
        self.kappa_abs = Parameter(**STANDARD_PARAMETERS.kappa_abs)

        # TODO: Add opacity weights
        # self.opacity_weights = Parameter()

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
        self._stellar_angular_radius = distance_to_angular(
            self.eff_radius(), self.dist()
        )
        return self._stellar_angular_radius

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
        rin, rout = self.rin().value, self.rout().value
        unit = self.rin().unit
        if OPTIONS.model.gridtype == "linear":
            radius = np.linspace(rin, rout, dim)
        else:
            radius = np.logspace(np.log10(rin), np.log10(rout), dim)
        radius *= unit
        return radius.astype(OPTIONS.data.dtype.real)

    def get_opacity(self, wavelength: u.um) -> u.cm**2 / u.g:
        """Set the opacity from wavelength."""
        if self.continuum_contribution:
            cont_weight = self.cont_weight()
            opacity = (1 - cont_weight) * self.kappa_abs(
                wavelength
            ) + cont_weight * self.kappa_cont(wavelength)
        else:
            opacity = self.kappa_abs(wavelength)

        opacity = opacity.astype(OPTIONS.data.dtype.real)
        if opacity.size == 1:
            return opacity.squeeze()

        shape = tuple(np.newaxis for _ in range(len(wavelength.shape) - 1))
        return opacity[(slice(None), *shape)]

    def compute_azimuthal_modulation(self, xx: u.mas, yy: u.mas) -> u.one:
        """Computes the azimuthal modulation."""
        if not self.asymmetric:
            return np.array([1])[:, np.newaxis]

        azimuthal_modulation = 1 + self.a() * np.cos(np.arctan2(yy, xx) - self.phi())
        return azimuthal_modulation.astype(OPTIONS.data.dtype.real)

    def compute_temperature(self, radius: u.mas) -> u.K:
        """Computes a 1D-temperature profile."""
        if self.r0.value != 0:
            reference_radius = self.r0()
        else:
            reference_radius = distance_to_angular(
                OPTIONS.model.reference_radius, self.dist()
            )

        if self.const_temperature:
            temperature = (
                np.sqrt(self.stellar_radius_angular / (2.0 * radius)) * self.eff_temp()
            )
        else:
            temperature = self.inner_temp() * (radius / reference_radius) ** (-self.q())
        return temperature.astype(OPTIONS.data.dtype.real)

    def compute_surface_density(self, radius: u.mas) -> u.one:
        """Computes a 1D-surface density profile."""
        if self.r0.value != 0:
            reference_radius = self.r0()
        else:
            reference_radius = distance_to_angular(
                OPTIONS.model.reference_radius, self.dist()
            )

        surface_density = self.inner_sigma() * (radius / reference_radius) ** (
            -self.p()
        )
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

    def compute_brightness(self, radius: u.mas, wavelength: u.um) -> u.Jy:
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

    # TODO: Think of a way to implement higher orders of modulation
    def compute_modulation(
        self,
        radius: u.mas,
        brightness: u.erg / (u.rad**2 * u.s * u.Hz),
        baselines: 1 / u.rad,
        baseline_angles: u.rad,
    ) -> u.Jy:
        """The azimuthal modulation as it appears in the hankel transform.

        The results of the modulation is flux in Jansky.

        Parameters
        ----------
        radius : astropy.units.mas
            The radius.
        brightness_profile : astropy.units.erg/(u.rad**2*u.s*u.Hz)
            The brightness profile.
        baselines : 1/astropy.units.rad
            The deprojected baselines.
        baseline_angles : astropy.units.rad
            The deprojected baseline angles.
        """
        if not self.asymmetric:
            return np.array([])

        angle_diff = baseline_angles - self.phi().to(u.rad)
        order = np.arange(1, OPTIONS.model.modulation + 1)[np.newaxis, np.newaxis, :]
        integrand = radius * brightness[:, np.newaxis, ...]
        bessel_factor = 2 * np.pi * radius.value * baselines.value[..., np.newaxis, :]

        if len(baseline_angles.shape) == 4:
            order = order[..., np.newaxis, :]

        factor = (-1j) ** order * self.a() * np.cos(order * angle_diff)
        integration = (
            2
            * np.pi
            * np.trapz(integrand * jv(order[..., np.newaxis], bessel_factor), radius)
        )
        return u.Quantity(factor * integration, unit=u.Jy).astype(
            OPTIONS.data.dtype.complex
        )

    def flux_func(self, wavelength: u.um) -> np.ndarray:
        """Computes the total flux from the hankel transformation."""
        radius = self.compute_internal_grid()
        brightness_profile = self.compute_brightness(radius, wavelength[:, np.newaxis])
        # TODO: Check if a factor of 2*np.pi is required in front of the integration
        integrand = 2 * np.pi * radius * brightness_profile
        flux = self.inc() * np.trapz(integrand, radius).to(u.Jy)
        return flux.value.reshape((flux.shape[0], 1)).astype(OPTIONS.data.dtype.real)

    def vis_func(
        self, baselines: 1 / u.rad, baseline_angles: u.rad, wavelength: u.um, **kwargs
    ) -> np.ndarray:
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
        radius = self.compute_internal_grid()
        brightness = self.compute_brightness(radius, wavelength)
        brightness = brightness[:, np.newaxis, :]

        radius = radius.to(u.rad)
        bessel_factor = 2 * np.pi * radius * baselines

        # TODO: Check if a factor of 2*np.pi is required in front of the integration
        vis = self.inc() * np.trapz(radius * brightness * j0(bessel_factor), radius).to(u.Jy)
        vis_mod = self.compute_modulation(
            radius, brightness, baselines, baseline_angles)

        if vis_mod.size != 0:
            vis = vis.astype(OPTIONS.data.dtype.complex) + vis_mod.sum(-1)

        return vis.value.astype(OPTIONS.data.dtype.complex)

    def image_func(self, xx: u.mas, yy: u.mas,
                   pixel_size: u.mas, wavelength: u.um) -> np.ndarray:
        """Computes the image."""
        radius = np.hypot(xx, yy)
        radial_profile = np.logical_and(radius >= self.rin(), radius <= self.rout())
        azimuthal_modulation = self.compute_azimuthal_modulation(xx, yy)
        azimuthal_modulation = azimuthal_modulation[np.newaxis, ...]
        brightness = (self.compute_brightness(radius, wavelength).to(
            u.erg / (u.cm**2 * u.rad**2 * u.s * u.Hz)) * pixel_size.to(u.rad) ** 2)
        return brightness.to(u.Jy).value * radial_profile * azimuthal_modulation


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
    elliptic = True
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
        if self.has_ring:
            self.hlr = np.sqrt(10 ** (2 * self.la()) / (1 + 10 ** (-2 * self.lkr())))
            self.rin = np.sqrt(10 ** (2 * self.la()) / (1 + 10 ** (2 * self.lkr())))
        else:
            self.hlr = 10 ** self.la()
        
        if self.is_gauss_lor:
            self.comp = GaussLorentzian(
                flor=self.flor, hlr=self.hlr, inc=self.inc, pa=self.pa)
        else:
            self.comp = Gaussian(hlr=self.hlr, inc=self.inc, pa=self.pa)

        if self.has_ring:
            self.ring = Ring(rin=self.rin, a=self.a, inc=self.inc,
                             pa=self.pa, phi=self.phi, asymmetric=True)

    def vis_func(self, baselines: 1 / u.rad, baseline_angles: u.rad,
                 wavelength: u.um, **kwargs) -> np.ndarray:
        fh = 1 - (self.fs() + self.fc())
        try:
            ks = self.ks(wavelength)[:, np.newaxis, np.newaxis]
        except TypeError:
            ks = self.ks(wavelength)[np.newaxis, np.newaxis]

        if len(baselines.shape) == 4:
            ks = ks[..., np.newaxis]

        if self.wl0() == 0:
            self.wl0.value = np.mean(wavelength)

        wavelength_ratio = self.wl0() / wavelength[..., np.newaxis]
        vis_star = self.fs() * wavelength_ratio**ks
        divisor = (fh + self.fs()) * wavelength_ratio**ks \
            + self.fc() * wavelength_ratio ** self.kc()

        vis_comp = self.comp.vis_func(
            baselines, baseline_angles, wavelength, **kwargs)
        if self.has_ring:
            vis_ring = self.ring.vis_func(
                baselines, baseline_angles, wavelength, **kwargs)
            vis_comp *= vis_ring

        vis_comp = self.fc() * vis_comp * wavelength_ratio ** self.kc()
        vis = (vis_star + vis_comp) / divisor
        return vis.value.astype(OPTIONS.data.dtype.complex)

    def image_func(self, xx: u.mas, yy: u.mas,
                   pixel_size: u.mas, wavelength: u.um) -> np.ndarray:
        """Computes the image."""
        fh = 1 - (self.fs() + self.fc())
        image = self.comp.image_func(xx, yy, pixel_size, wavelength)
        if self.has_ring:
            image_ring = self.ring.image_func(xx, yy, pixel_size, wavelength)
            image = fftconvolve(image, image_ring, mode="same")
            
        image *= self.fc()
        pt = PointSource(inc=self.inc, pa=self.pa)
        image += pt.image_func(xx, yy, pixel_size, wavelength) * self.fs() + fh
        return image.astype(OPTIONS.data.dtype.real)

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


def assemble_components(
    parameters: Dict[str, Dict], shared_params: Optional[Dict[str, Parameter]] = None
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
