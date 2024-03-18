import sys
from typing import Tuple, Optional, Dict, List
import astropy.units as u
import numpy as np
from astropy.modeling.models import BlackBody
from scipy.special import j0, j1, jv

from .component import Component, Convolver
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

    def vis_func(self, baselines: 1/u.rad, baseline_angles: u.rad,
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
        new_shape = (-1,) + (1,) * (len(baselines.shape)-1)
        vis = np.tile(self.flux_func(wavelength).reshape(new_shape), baselines.shape[1:])
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
        centre = xx.shape[0]//2
        star_flux = (self.compute_flux(wavelength)/4)[..., np.newaxis]
        image[:, centre-1:centre+1, centre-1:centre+1] = star_flux
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
            self.eff_radius(), self.dist())
        return self._stellar_angular_radius

    def flux_func(self, wavelength: u.um) -> np.ndarray:
        """Computes the flux of the star."""
        if self.f.value is not None:
            stellar_flux = self.f(wavelength)
        else:
            plancks_law = BlackBody(temperature=self.eff_temp())
            spectral_radiance = plancks_law(wavelength).to(
                u.erg/(u.cm**2*u.Hz*u.s*u.rad**2))
            stellar_flux = np.pi*(spectral_radiance
                                  * self.stellar_radius_angular**2).to(u.Jy)
        return stellar_flux.value.reshape((wavelength.size, 1))

    def vis_func(self, baselines: 1/u.rad, baseline_angles: u.rad,
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
        new_shape = (-1,) + (1,) * (len(baselines.shape)-1)
        vis = np.tile(self.flux_func(wavelength).reshape(new_shape), baselines.shape[1:])
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
        centre = xx.shape[0]//2
        star_flux = (self.compute_flux(wavelength)/4)[..., np.newaxis]
        image[:, centre-1:centre+1, centre-1:centre+1] = star_flux
        return image


class Ring(Component):
    name = "Ring"
    shortname = "Ring"
    description = "A simple ring."
    _thin = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rin = Parameter(**STANDARD_PARAMETERS.rin)
        self.width = Parameter(**STANDARD_PARAMETERS.width)
        self.eval(**kwargs)

    @property
    def thin(self) -> bool:
        """Gets if the component is elliptic."""
        return self._thin

    @thin.setter
    def thin(self, value: bool) -> None:
        """Sets the position angle and the parameters to free or false
        if elliptic is set."""
        self.width.free = not value
        self._thin = value

    def vis_func(self, baselines: 1/u.rad, baseline_angles: u.rad,
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
        if self.thin:
            vis = j0(2*np.pi*self.rin().to(u.rad)*baselines).astype(complex)
            if self.asymmetric:
                vis += -1j*self.a()*np.cos(1j*(baseline_angles-self.phi().to(u.rad)))\
                    * j1(2*np.pi*self.rin().to(u.rad)*baselines)
        else:
            radius = np.linspace(self.rin(), self.rin()+self.width(), self.dim())
            vis = (np.trapz(j0(2*np.pi*radius.to(u.rad)*baselines), radius)/self.width()).astype(complex)
            if self.asymmetric:
                vis += -1j*self.a()*np.cos(1j*(baseline_angles-self.phi().to(u.rad)))\
                    * np.trapz(j1(2*np.pi*radius.to(u.rad)*baselines), radius)/self.width()
        return (self.fr()*vis).value.astype(OPTIONS.data.dtype.complex)

    def image_func(self, xx: u.mas, yy: u.mas, pixel_size: u.mas, wavelength: u.um) -> np.ndarray:
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
        radial_profile = np.logical_and(
                radius >= self.rin(), radius <= self.rin()+self.width())
        return radial_profile.astype(OPTIONS.data.dtype.real)


class UniformDisk(Component):
    name = "Uniform Disk"
    shortname = "UniformDisk"
    description = "A uniform disk."

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.diam = Parameter(**STANDARD_PARAMETERS.diam)
        self.eval(**kwargs)

    def vis_func(self, baselines: 1/u.rad, baseline_angles: u.rad,
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
        vis = (2*j1(np.pi*self.diam().to(u.rad)*baselines)/(np.pi*self.diam().to(u.rad)*baselines))
        return vis.value.astype(OPTIONS.data.dtype.complex)


class Gaussian(Component):
    name = "Gaussian"
    shortname = "Gaussian"
    description = "A simple 2D Gaussian."

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fwhm = Parameter(**STANDARD_PARAMETERS.fwhm)
        self.eval(**kwargs)

    def vis_func(self, baselines: 1/u.rad, baseline_angles: u.rad,
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
        vis = np.exp(-(np.pi*baselines*self.fwhm().to(u.rad))**2/(4*np.log(2)))
        return (self.fr()*vis).value.astype(OPTIONS.data.dtype.complex)


class Lorentzian(Component):
    name = "Lorentzian"
    shortname = "Lorentzian"
    description = "A simple 2D Lorentzian."

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fwhm = Parameter(**STANDARD_PARAMETERS.fwhm)
        self.eval(**kwargs)

    def vis_func(self, baselines: 1/u.rad, baseline_angles: u.rad,
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
        vis = np.exp(-np.pi*baselines*self.fwhm().to(u.rad)/np.sqrt(3))
        return (self.fr()*vis).value.astype(OPTIONS.data.dtype.complex)


class GaussLorentzian(Component):
    name = "Gauss-Lorentzian"
    shortname = "GaussLorentzian"
    description = "A simple 2D Gaussian combined with a Lorentzian."

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.flor = Parameter(**STANDARD_PARAMETERS.fr)
        self.flor.free = True
        self.fwhm = Parameter(**STANDARD_PARAMETERS.fwhm)
        self.eval(**kwargs)

    def vis_func(self, baselines: 1/u.rad, baseline_angles: u.rad,
                 wavelength: u.um, **kwargs) -> np.ndarray:
        gauss = Gaussian(fwhm=self.fwhm, inc=self.inc, pa=self.pa)
        lor = Lorentzian(fwhm=self.fwhm, inc=self.inc, pa=self.pa)
        vis_gauss = gauss.vis_func(baselines, baseline_angles, wavelength, **kwargs)
        vis_lor = lor.vis_func(baselines, baseline_angles, wavelength, **kwargs)
        vis = (1-self.flor())*vis_gauss + self.flor()*vis_lor
        return (self.fr()*vis).value.astype(OPTIONS.data.dtype.complex)


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
    name = "Hankel Component"
    shortname = "HankComp"
    description = "This defines the analytical hankel transformation."
    elliptic = True
    _optically_thick = False
    _const_temperature = False
    _continuum_contribution = True

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
            self.eff_radius(), self.dist())
        return self._stellar_angular_radius

    def compute_internal_grid(self, dim: int) -> u.mas:
        """Computes the model grid.

        Parameters
        ----------
        dim : float

        Returns
        -------
        radial_grid : astropy.units.mas
            A one dimensional linear or logarithmic grid.
        """
        rin, rout = self.rin(), self.rout()
        if OPTIONS.model.gridtype == "linear":
            radius = np.linspace(rin.value, rout.value, dim)*self.rin.unit
        else:
            radius = np.logspace(np.log10(rin.value),
                                 np.log10(rout.value), dim)*self.rin.unit
        return radius.astype(OPTIONS.data.dtype.real)

    def get_opacity(self, wavelength: u.um) -> u.cm**2/u.g:
        """Set the opacity from wavelength."""
        if self.continuum_contribution:
            cont_weight = self.cont_weight()
            opacity = (1-cont_weight)*self.kappa_abs(wavelength)\
                + cont_weight*self.kappa_cont(wavelength)
        else:
            opacity = self.kappa_abs(wavelength)

        opacity = opacity.astype(OPTIONS.data.dtype.real)
        if opacity.size == 1:
            return opacity.squeeze()

        shape = tuple(np.newaxis for _ in range(len(wavelength.shape)-1))
        return opacity[(slice(None), *shape)]

    # TODO: Maybe to move to component
    def compute_azimuthal_modulation(self, xx: u.mas, yy: u.mas) -> u.one:
        """Computes the azimuthal modulation."""
        if not self.asymmetric:
            return np.array([1])[:, np.newaxis]

        azimuthal_modulation = (1+self.a()\
                * np.cos(np.arctan2(yy, xx)-self.phi()))
        return azimuthal_modulation.astype(OPTIONS.data.dtype.real)

    def compute_temperature(self, radius: u.mas) -> u.K:
        """Computes a 1D-temperature profile."""
        if self.r0.value != 0:
            reference_radius = self.r0()
        else:
            reference_radius = distance_to_angular(
                    OPTIONS.model.reference_radius, self.dist())

        if self.const_temperature:
            temperature = np.sqrt(self.stellar_radius_angular/(2.0*radius))\
                    * self.eff_temp()
        else:
            temperature = self.inner_temp()\
                * (radius/reference_radius)**(-self.q())
        return temperature.astype(OPTIONS.data.dtype.real)

    def compute_surface_density(self, radius: u.mas) -> u.one:
        """Computes a 1D-surface density profile."""
        if self.r0.value != 0:
            reference_radius = self.r0()
        else:
            reference_radius = distance_to_angular(
                    OPTIONS.model.reference_radius, self.dist())

        surface_density = self.inner_sigma()\
            * (radius/reference_radius)**(-self.p())
        return surface_density.astype(OPTIONS.data.dtype.real)

    def compute_emissivity(self, radius: u.mas, wavelength: u.um) -> u.one:
        """Computes a 1D-emissivity profile."""
        if wavelength.shape == ():
            wavelength.reshape((wavelength.size,))

        if self.optically_thick:
            return np.array([1])[:, np.newaxis]

        surface_density = self.compute_surface_density(radius)
        optical_depth = surface_density*self.get_opacity(wavelength)
        emissivity = (1-np.exp(-optical_depth/self.inc()))
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
        return (brightness*emissivity).astype(OPTIONS.data.dtype.real)

    # TODO: Think of a way to implement higher orders of modulation
    # TODO: Check all the broadcasting
    def compute_hankel_modulation(self, radius: u.mas,
                                  brightness: u.erg/(u.rad**2*u.s*u.Hz),
                                  baselines: 1/u.rad, baseline_angles: u.rad) -> u.Jy:
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

        angle_diff = baseline_angles-self.phi().to(u.rad)
        order = np.arange(1, OPTIONS.model.modulation+1)[np.newaxis, np.newaxis, :]
        integrand = radius*brightness[:, np.newaxis, ...]
        bessel_factor = radius.value*baselines.value[..., np.newaxis, :]

        if len(baseline_angles.shape) == 4:
            order = order[..., np.newaxis, :]

        factor = (-1j)**order*self.a()*np.cos(order*angle_diff)
        integration = 2*np.pi*np.trapz(integrand * jv(
                    order[..., np.newaxis], 2.*np.pi*bessel_factor), radius)
        return u.Quantity(factor*integration, unit=u.Jy)

    # TODO: Check all the broadcasting
    def compute_hankel_transform(self, radius: u.mas, baselines: 1/u.rad,
                                 baseline_angles: u.rad, wavelength: u.um) -> Tuple[u.Quantity, u.Quantity]:
        """Executes the hankel transform and returns the correlated fluxes
        and their modulations.

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
        correlated_fluxes : astropy.units.Jy
            The visibilities.
        modulations : astropy.units.Jy
        """
        brightness = self.compute_brightness(radius, wavelength)
        brightness = brightness[:, np.newaxis, :]

        radius = radius.to(u.rad)
        visibility = 2*self.inc()*np.pi*np.trapz(radius*brightness*j0(
            2.*np.pi*radius.value*baselines.value), radius).to(u.Jy)
        modulation = self.compute_hankel_modulation(
                radius, brightness, baselines, baseline_angles)
    
        return visibility.astype(OPTIONS.data.dtype.complex), \
            modulation.astype(OPTIONS.data.dtype.complex)

    def flux_func(self, wavelength: u.um) -> np.ndarray:
        """Computes the total flux from the hankel transformation."""
        radius = self.compute_internal_grid(self.dim())
        brightness_profile = self.compute_brightness(
                radius, wavelength[:, np.newaxis])
        flux = (2.*np.pi*self.inc()*np.trapz(
            radius*brightness_profile, radius).to(u.Jy))
        return flux.value.reshape((flux.shape[0], 1)).astype(OPTIONS.data.dtype.real)

    def vis_func(self, baselines: 1/u.rad, baseline_angles: u.rad,
                 wavelength: u.um, **kwargs) -> np.ndarray:
        """Computes the correlated fluxes via the hankel transformation."""
        radius = self.compute_internal_grid(self.dim())
        vis, vis_mod = self.compute_hankel_transform(
                radius, baselines, baseline_angles, wavelength, **kwargs)

        if vis_mod.size != 0:
            if len(baselines.shape) <= 3:
                vis += vis_mod.sum(-1)
            else:
                vis += np.concatenate(
                        (vis_mod[:, :2], np.conj(vis_mod[:, 2:])), axis=1).sum(-1)

        return vis.value.astype(OPTIONS.data.dtype.complex)

    def image_func(self, xx: u.mas, yy: u.mas,
                   pixel_size: u.mas, wavelength: u.um) -> np.ndarray:
        """Computes the image."""
        radius = np.hypot(xx, yy)
        radial_profile = np.logical_and(radius >= self.rin(), radius <= self.rout())
        azimuthal_modulation = self.compute_azimuthal_modulation(xx, yy)
        azimuthal_modulation = azimuthal_modulation[np.newaxis, ...]
        brightness = self.compute_brightness(
                radius, wavelength).to(u.erg/(u.cm**2*u.rad**2*u.s*u.Hz))\
            * pixel_size.to(u.rad)**2
        return brightness.to(u.Jy).value*radial_profile*azimuthal_modulation


class AsymmetricTempGradient(TempGradient):
    """An analytical implementation of an asymmetric temperature
    gradient."""
    name = "Asymmetric Continuum Grey Body"
    shortname = "AsymContinuumGreyBody"
    asymmetric = True


class GreyBody(TempGradient):
    """An analytical implementation of an asymmetric temperature
    gradient."""
    name = "Asymmetric Continuum Grey Body"
    shortname = "AsymContinuumGreyBody"
    const_temperature = True


class AsymmetricGreyBody(TempGradient):
    """An analytical implementation of an asymmetric temperature
    gradient."""
    name = "Asymmetric Continuum Grey Body"
    shortname = "AsymContinuumGreyBody"
    asymmetric = True
    const_temperature = True


# class PointRing(Component):
#     name = "PointDisk"
#     shortname = "PointDisk"

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.fs = Parameter(**STANDARD_PARAMETERS.fr)
#         self.fc = Parameter(**STANDARD_PARAMETERS.fr)
#         self.flor = Parameter(**STANDARD_PARAMETERS.fr)
#         self.fwhm = Parameter(**STANDARD_PARAMETERS.fwhm)
#         self.wl0 = Parameter(**STANDARD_PARAMETERS.wl)
#         self.ks = Parameter(**STANDARD_PARAMETERS.exp)
#         self.ks.free = False
#         self.kc = Parameter(**STANDARD_PARAMETERS.exp)

class StarHaloGaussLor(Component):
    """A star, a disk and a halo model as seen in Lazareff+2017."""
    name = "StarDiskHalo"
    shortname = "StarDiskHalo"
    ring = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fs = Parameter(**STANDARD_PARAMETERS.fr)
        self.fc = Parameter(**STANDARD_PARAMETERS.fr)
        self.flor = Parameter(**STANDARD_PARAMETERS.fr)
        self.rin = Parameter(**STANDARD_PARAMETERS.rin)
        self.fwhm = Parameter(**STANDARD_PARAMETERS.fwhm)
        self.wl0 = Parameter(**STANDARD_PARAMETERS.wl)
        self.ks = Parameter(**STANDARD_PARAMETERS.exp)
        self.ks.free = False
        self.kc = Parameter(**STANDARD_PARAMETERS.exp)
        self.eval(**kwargs)

    def vis_func(self, baselines: 1 / u.rad, baseline_angles: u.rad,
                 wavelength: u.um, **kwargs) -> np.ndarray:
        fh = 1 - self.fs() - self.fc()
        ks = self.ks(wavelength)[:, np.newaxis, np.newaxis]
        if len(baselines.shape) == 4:
            ks = ks[..., np.newaxis]
        wavelength_ratio = self.wl0()/wavelength[..., np.newaxis]
        vis_star = self.fs()*wavelength_ratio**ks
        divisor = (fh+self.fs())*wavelength_ratio**ks \
            + self.fc()*wavelength_ratio**self.kc

        gl = GaussLorentzian(flor=self.flor, fwhm=self.fwhm,
                             inc=self.inc, pa=self.pa)
        if self.ring:
            ring = Ring(rin=self.rin, a=self.a, inc=self.inc,
                        pa=self.pa, phi=self.phi, asymmetric=True)
            conv = Convolver(gl=gl, ring=ring)
            vis_disk = conv.vis_func(baselines, baseline_angles, wavelength, **kwargs)
        else:
            vis_disk = gl.vis_func(baselines, baseline_angles, wavelength, **kwargs)

        vis_comp = self.fc()*vis_disk*wavelength_ratio**self.kc()
        return ((vis_star+vis_comp)/divisor).value.astype(OPTIONS.data.dtype.complex)


class StarHaloRing(StarHaloGaussLor):
    """A star, a disk and a halo model as seen in Lazareff+2017."""
    name = "StarDiskHalo"
    shortname = "StarDiskHalo"
    ring = True


def assemble_components(
        parameters: Dict[str, Dict],
        shared_params: Optional[Dict[str, Parameter]] = None
        ) -> List[Component]:
    """Assembles a model from a dictionary of parameters."""
    components = []
    for (component, params) in parameters:
        comp = getattr(sys.modules[__name__], component)
        components.append(comp(**params, **shared_params,
                               **OPTIONS.model.constant_params))
    return components


