import sys
from typing import Tuple, Optional, Dict, List

import astropy.units as u
import numpy as np
from astropy.modeling.models import BlackBody
from scipy.special import j0, jv

from .component import Component
from .parameter import Parameter
from .options import STANDARD_PARAMETERS, OPTIONS
from .utils import distance_to_angular, compute_effective_baselines


class Star(Component):
    """Star defined as an analytical component.

    Parameters
    ----------
    x : int
        x pos of the component [mas].
    y : int
        y pos of the component [mas].
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
    params : dict of Parameter
    stellar_radius_angular : u.mas
    _image : numpy.ndarray
    """
    name = "Star"
    shortname = "St"
    description = "The flux of a star."

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._stellar_angular_radius = None

        self.params["f"] = Parameter(**STANDARD_PARAMETERS["f"])
        self.params["dist"] = Parameter(**STANDARD_PARAMETERS["dist"])
        self.params["eff_temp"] = Parameter(**STANDARD_PARAMETERS["eff_temp"])
        self.params["eff_radius"] = Parameter(**STANDARD_PARAMETERS["eff_radius"])
        self._eval(**kwargs)

    @property
    def stellar_radius_angular(self) -> u.mas:
        r"""Calculates the parallax from the stellar radius and the distance to
        the object.

        Returns
        -------
        stellar_radius_angular : astropy.units.mas
            The parallax of the stellar radius.
        """
        self._stellar_angular_radius = distance_to_angular(
            self.params["eff_radius"](), self.params["dist"]())
        return self._stellar_angular_radius

    def flux_func(self, wavelength: u.um) -> u.Jy:
        """Calculates the flux of the star."""
        if self.params["f"].value is not None:
            stellar_flux = self.params["f"](wavelength)
        else:
            plancks_law = BlackBody(temperature=self.params["eff_temp"]())
            spectral_radiance = plancks_law(wavelength).to(
                u.erg/(u.cm**2*u.Hz*u.s*u.rad**2))
            stellar_flux = np.pi*(spectral_radiance
                                  * self.stellar_radius_angular**2).to(u.Jy)
        return stellar_flux.reshape((wavelength.size, 1))

    def image_func(self, xx: u.mas, yy: u.mas,
                   pixel_size: u.mas, wavelength: u.m = None) -> u.Jy:
        """Calculates the image from a 2D grid.

        Parameters
        ----------
        xx : u.mas
        yy : u.mas
        wavelength : u.m

        Returns
        -------
        image : astropy.units.Jy
        """
        image = np.zeros((wavelength.size, *xx.shape))*u.Jy
        centre = xx.shape[0]//2
        star_flux = (self.compute_flux(wavelength)/4)[..., np.newaxis]
        image[:, centre-1:centre+1, centre-1:centre+1] = star_flux
        return image


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
    asymmetric = False
    optically_thick = False
    const_temperature = False
    continuum_contribution = True

    def __init__(self, **kwargs):
        """The class's constructor."""
        super().__init__(**kwargs)
        self._stellar_angular_radius = None

        self.params["dist"] = Parameter(**STANDARD_PARAMETERS["dist"])
        self.params["eff_temp"] = Parameter(**STANDARD_PARAMETERS["eff_temp"])
        self.params["eff_radius"] = Parameter(**STANDARD_PARAMETERS["eff_radius"])

        self.params["r0"] = Parameter(**STANDARD_PARAMETERS["r0"])
        self.params["rin"] = Parameter(**STANDARD_PARAMETERS["rin"])
        self.params["rout"] = Parameter(**STANDARD_PARAMETERS["rout"])

        self.params["a"] = Parameter(**STANDARD_PARAMETERS["a"])
        self.params["phi"] = Parameter(**STANDARD_PARAMETERS["phi"])

        self.params["q"] = Parameter(**STANDARD_PARAMETERS["q"])
        self.params["inner_temp"] = Parameter(**STANDARD_PARAMETERS["inner_temp"])

        self.params["p"] = Parameter(**STANDARD_PARAMETERS["p"])
        self.params["inner_sigma"] = Parameter(**STANDARD_PARAMETERS["inner_sigma"])
        self.params["kappa_abs"] = Parameter(**STANDARD_PARAMETERS["kappa_abs"])
        self.params["cont_weight"] = Parameter(**STANDARD_PARAMETERS["cont_weight"])
        self.params["kappa_cont"] = Parameter(**STANDARD_PARAMETERS["kappa_cont"])

        if self.const_temperature:
            self.params["q"].free = False
            self.params["inner_temp"].free = False

        if not self.asymmetric:
            self.params["a"].free = False
            self.params["phi"].free = False

        if not self.continuum_contribution:
            self.params["cont_weight"].free = False
        self._eval(**kwargs)

    @property
    def stellar_radius_angular(self) -> u.mas:
        r"""Calculates the parallax from the stellar radius and the distance to
        the object.

        Returns
        -------
        stellar_radius_angular : astropy.units.mas
            The parallax of the stellar radius.
        """
        self._stellar_angular_radius = distance_to_angular(
            self.params["eff_radius"](), self.params["dist"]())
        return self._stellar_angular_radius

    def compute_internal_grid(self, dim: int) -> u.mas:
        """Calculates the model grid.

        Parameters
        ----------
        dim : float

        Returns
        -------
        radial_grid : astropy.units.mas
            A one dimensional linear or logarithmic grid.
        """
        rin, rout = self.params["rin"](), self.params["rout"]()
        if OPTIONS.model.gridtype == "linear":
            radius = np.linspace(rin.value, rout.value, dim)*self.params["rin"].unit
        else:
            radius = np.logspace(np.log10(rin.value),
                                 np.log10(rout.value), dim)*self.params["rin"].unit
        return radius.astype(OPTIONS.data.dtype.real)

    def get_opacity(self, wavelength: u.um) -> u.cm**2/u.g:
        """Set the opacity from wavelength."""
        if self.continuum_contribution:
            cont_weight = self.params["cont_weight"]()
            opacity = (1-cont_weight)*self.params["kappa_abs"](wavelength)\
                + cont_weight*self.params["kappa_cont"](wavelength)
        else:
            opacity = self.params["kappa_abs"](wavelength)

        opacity = opacity.astype(OPTIONS.data.dtype.real)
        if opacity.size == 1:
            return opacity.squeeze()

        shape = tuple(np.newaxis for _ in range(len(wavelength.shape)-1))
        return opacity[(slice(None), *shape)]

    def compute_azimuthal_modulation(self, xx: u.mas, yy: u.mas) -> u.one:
        """Calculates the azimuthal modulation."""
        if not self.asymmetric:
            return np.array([1])[:, np.newaxis]

        azimuthal_modulation = (1+self.params["a"]()\
                * np.cos(np.arctan2(yy, xx)-self.params["phi"]()))
        return azimuthal_modulation.astype(OPTIONS.data.dtype.real)

    def compute_temperature(self, radius: u.mas) -> u.K:
        """Calculates a 1D-temperature profile."""
        if self.params["r0"].value != 0:
            reference_radius = self.params["r0"]()
        else:
            reference_radius = distance_to_angular(
                    OPTIONS.model.reference_radius, self.params["dist"]())

        if self.const_temperature:
            temperature = np.sqrt(self.stellar_radius_angular/(2.0*radius))\
                    * self.params["eff_temp"]()
        else:
            temperature = self.params["inner_temp"]()\
                * (radius/reference_radius)**(-self.params["q"]())
        return temperature.astype(OPTIONS.data.dtype.real)

    def compute_surface_density(self, radius: u.mas) -> u.one:
        """Calculates a 1D-surface density profile."""
        if self.params["r0"].value != 0:
            reference_radius = self.params["r0"]()
        else:
            reference_radius = distance_to_angular(
                    OPTIONS.model.reference_radius, self.params["dist"]())

        surface_density = self.params["inner_sigma"]()\
            * (radius/reference_radius)**(-self.params["p"]())
        return surface_density.astype(OPTIONS.data.dtype.real)

    def compute_emissivity(self, radius: u.mas, wavelength: u.um) -> u.one:
        """Calculates a 1D-emissivity profile."""
        if wavelength.shape == ():
            wavelength.reshape((wavelength.size,))

        if self.optically_thick:
            return np.array([1])[:, np.newaxis]

        surface_density = self.compute_surface_density(radius)
        optical_depth = surface_density*self.get_opacity(wavelength)
        emissivity = (1-np.exp(-optical_depth/self.params["elong"]()))
        return emissivity.astype(OPTIONS.data.dtype.real)

    def compute_brightness(self, radius: u.mas, wavelength: u.um) -> u.Jy:
        """Calculates a 1D-brightness profile from a dust-surface density- and
        temperature profile.

        Parameters
        ----------
        wl : astropy.units.um
            Wavelengths.

        Returns
        -------
        brightness_profile : astropy.units.Jy
        """
        if wavelength.shape == ():
            wavelength.reshape((wavelength.size,))

        temperature = self.compute_temperature(radius)
        brightness = BlackBody(temperature)(wavelength)
        emissivity = self.compute_emissivity(radius, wavelength)
        return (brightness*emissivity).astype(OPTIONS.data.dtype.real)

    # TODO: Think of a way to implement higher orders of modulation
    # TODO: Check all the broadcasting
    def compute_hankel_modulation(self, radius: u.mas,
                                  brightness_profile: u.erg/(u.rad**2*u.s*u.Hz),
                                  baselines: u.rad, baseline_angles: u.rad) -> u.Jy:
        """The azimuthal modulation as it appears in the hankel transform.

        The results of the modulation is flux in Jansky.

        Parameters
        ----------
        radius : astropy.units.mas
            The radius.
        brightness_profile : astropy.units.erg/(u.rad**2*u.s*u.Hz)
            The brightness profile.
        baselines : astropy.units.rad
            The baseline.
        baseline_angles : astropy.units.rad
            The baseline angle.
        """
        if not self.asymmetric:
            return np.array([])

        angle_diff = baseline_angles-self.params["phi"]().to(u.rad)
        order = np.arange(1, OPTIONS.model.modulation+1)[np.newaxis, np.newaxis, :]
        integrand = radius*brightness_profile[:, np.newaxis, ...]
        bessel_factor = radius.value*baselines.value[..., np.newaxis, :]

        if len(baseline_angles.shape) == 4:
            order = order[..., np.newaxis, :]

        factor = (-1j)**order*self.params["a"]()*np.cos(order*(angle_diff))
        integration = 2*np.pi*np.trapz(integrand * jv(
                    order[..., np.newaxis], 2.*np.pi*bessel_factor), radius)
        return u.Quantity(factor*integration, unit=u.Jy)

    # TODO: Check all the broadcasting
    def compute_hankel_transform(self, radius: u.mas, ucoord: u.m,
                                 vcoord: u.m, wavelength: u.um) -> Tuple[u.Quantity, u.Quantity]:
        """Executes the hankel transform and returns the correlated fluxes
        and their modulations.

        Parameters
        ----------
        radius : astropy.units.mas
            The radius.
        ucoord : astropy.units.m
            The u coordinates.
        vcoord : astropy.units.m
            The v coordinates.
        wavelength : astropy.units.um
            The wavelengths.

        Returns
        -------
        correlated_fluxes : astropy.units.Jy
            The visibilities.
        modulations : astropy.units.Jy
        """
        compression = self.params["elong"]()
        baselines, baseline_angles = compute_effective_baselines(
                ucoord, vcoord, compression, self.params["pa"]())

        wavelength = wavelength[:, np.newaxis]
        brightness = self.compute_brightness(radius, wavelength)
        radius = radius.to(u.rad)

        wavelength = wavelength.to(u.m)
        brightness = brightness[:, np.newaxis, :]
        if len(ucoord.shape) == 1:
            baselines = (baselines/wavelength).value*u.rad
            baselines = baselines[..., np.newaxis]
            baseline_angles = baseline_angles[np.newaxis, :, np.newaxis]
        else:
            wavelength = wavelength[..., np.newaxis]
            baselines = (baselines[np.newaxis, ...]/wavelength).value*u.rad
            baselines = baselines[..., np.newaxis]
            baseline_angles = baseline_angles[np.newaxis, ..., np.newaxis]
            brightness = brightness[..., np.newaxis, :]

        visibility = 2*compression*np.pi*np.trapz(radius*brightness*j0(
            2.*np.pi*radius.value*baselines.value), radius).to(u.Jy)
        modulation = self.compute_hankel_modulation(
                radius, brightness, baselines, baseline_angles)

        return visibility.astype(OPTIONS.data.dtype.complex), \
            modulation.astype(OPTIONS.data.dtype.complex)

    def flux_func(self, wavelength: u.um) -> u.Jy:
        """Calculates the total flux from the hankel transformation."""
        compression = self.params["elong"]()
        radius = self.compute_internal_grid(self.params["dim"]())
        brightness_profile = self.compute_brightness(
                radius, wavelength[:, np.newaxis])
        flux = (2.*np.pi*np.trapz(
            radius*compression*brightness_profile, radius).to(u.Jy)).value
        return flux.reshape((flux.shape[0], 1))

    def vis_func(self, ucoord: u.m, vcoord: u.m,
                 wavelength: u.um, **kwargs) -> np.ndarray:
        """Calculates the correlated fluxes via the hankel transformation."""
        radius = self.compute_internal_grid(self.params["dim"]())
        vis, vis_mod = self.compute_hankel_transform(
                radius, ucoord, vcoord, wavelength, **kwargs)
        if vis_mod.size != 0:
            vis += vis_mod.sum(-1)
        return vis.value

    def t3_func(self, ucoord: u.m, vcoord: u.m,
                wavelength: u.um, **kwargs) -> np.ndarray:
        """Calculates the closure phases via hankel transformation."""
        radius = self.compute_internal_grid(self.params["dim"]())
        vis, vis_mod = self.compute_hankel_transform(
                radius, ucoord, vcoord, wavelength, **kwargs)
        if vis_mod.size != 0:
            vis += np.concatenate(
                    (vis_mod[:, :2], np.conj(vis_mod[:, 2:])), axis=1).sum(-1)
        return vis.value

    def image_func(self, xx: u.mas, yy: u.mas,
                   pixel_size: u.mas, wavelength: u.um) -> u.Jy:
        """Calculates the image."""
        radius = np.hypot(xx, yy)
        radial_profile = np.logical_and(radius >= self.params["rin"](),
                                        radius <= self.params["rout"]())
        azimuthal_modulation = self.compute_azimuthal_modulation(xx, yy)
        azimuthal_modulation = azimuthal_modulation[np.newaxis, ...]
        brightness = self.compute_brightness(
                radius, wavelength).to(u.erg/(u.cm**2*u.rad**2*u.s*u.Hz))\
            * pixel_size.to(u.rad)**2
        return brightness.to(u.Jy)*radial_profile*azimuthal_modulation


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
