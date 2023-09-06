import sys
from typing import Optional, Dict, List

import astropy.units as u
import numpy as np
from astropy.modeling import models

from .component import Component, AnalyticalComponent, NumericalComponent
from .parameter import STANDARD_PARAMETERS, Parameter
from .options import OPTIONS
from ._spectral_cy import const_temperature,\
    temperature_power_law, azimuthal_modulation,\
    optical_thickness, surface_density_profile, intensity
from .utils import distance_to_angular,\
    get_new_dimension, rebin_image, pad_image


class Star(AnalyticalComponent):
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
        self._image, self._visibility = None, None
        self._stellar_angular_radius = None

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
        if self._stellar_angular_radius is None:
            self._stellar_angular_radius = distance_to_angular(
                self.params["eff_radius"](), self.params["dist"]())
        return self._stellar_angular_radius

    def caclculate_stellar_flux(self, wavelength: u.um) -> u.Jy:
        """Calculates the flux of the star."""
        plancks_law = models.BlackBody(temperature=self.params["eff_temp"]())
        spectral_radiance = plancks_law(wavelength.to(u.m)).to(
            u.erg/(u.cm**2*u.Hz*u.s*u.rad**2))
        return (spectral_radiance*self.stellar_radius_angular**2).to(u.Jy)

    def _image_function(self, xx: u.mas, yy: u.mas,
                        wavelength: Optional[u.Quantity[u.m]] = None,
                        ) -> Optional[u.Quantity]:
        """Calculates the image from a 2D grid.

        Parameters
        ----------
        xx : u.mas
        yy : u.mas
        wavelength : u.m, optional

        Returns
        -------
        image : astropy.units.Quantity, optional
        """
        if self._image is None:
            self._image = np.zeros(xx.shape)
            val = np.abs(xx)+np.abs(yy)
            index = np.unravel_index(np.argmin(val), np.shape(val))
            self._image[index] = 1
        return self._image*self.caclculate_stellar_flux(wavelength)

    def _visibility_function(self,
                             wavelength: Optional[u.Quantity[u.um]] = None
                             ) -> np.ndarray:
        """The component's _visibility_function."""
        if self._visibility is None:
            dim = get_new_dimension(self.params["dim"](),
                                    OPTIONS["fourier.binning"],
                                    OPTIONS["fourier.padding"])
            self._visibility = np.ones((dim, dim))

        return self._visibility*self.caclculate_stellar_flux(wavelength)


class TemperatureGradient(NumericalComponent):
    """A ring defined by a radial temperature profile in r^q
    that is multiplied by an azimuthal modulation.
    and an asymmetric radial dust surface density profile in r^p.

    Parameters
    ----------
    rin : float
        Inner radius of the disk [mas].
    rout : float
        Outer radius of the disk [mas].
    inner_temp : float
        Inner radius temperature [K].
    inner_sigma : float
        Inner surface density [g/cm^2].
    a : float
        Azimuthal modulation amplitude.
    phi : float
        Azimuthal modulation angle [deg].
    q : float
        Power-law exponent for the temperature profile.
    p : float
        Power-law exponent for the dust surface density profile.
    kappa_abs : float
        Dust mass absorption coefficient [cm^2g^-1].
    dist : float
        Distance of the star [pc].
    pa : float
        Positional angle [deg].
    elong : float
        Elongation of the disk [dimensionless].
    dim : float
        Dimension of the image [px].

    Attributes
    ----------
    params : dict with keys of str and values of Parameter
        Dictionary of parameters.
    """
    name = "Asymmetric Temperature Gradient"
    shortname = "AsymTempGrad"
    elliptic = True
    asymmetric_image = False
    asymmetric_surface_density = False
    optically_thick = False
    const_temperature = False
    continuum_contribution = False

    def __init__(self, **kwargs):
        """The class's constructor."""
        super().__init__(**kwargs)
        self.radius = None
        self._stellar_angular_radius = None

        self.params["dist"] = Parameter(**STANDARD_PARAMETERS["dist"])
        self.params["eff_temp"] = Parameter(**STANDARD_PARAMETERS["eff_temp"])
        self.params["eff_radius"] = Parameter(**STANDARD_PARAMETERS["eff_radius"])

        self.params["rin0"] = None

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

        if not self.asymmetric_image or not self.asymmetric_surface_density:
            self.params["a"].free = False
            self.params["phi"].free = False

        if self.continuum_contribution:
            self.params["cont_weight"].free = False
            self.params["kappa_cont"].free = False
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
        if self._stellar_angular_radius is None:
            self._stellar_angular_radius = distance_to_angular(
                self.params["eff_radius"](), self.params["dist"]())
        return self._stellar_angular_radius

    def _get_opacity(self, wavelength: u.um) -> u.cm**2/u.g:
        """Set the opacity from wavelength."""
        if self.continuum_contribution:
            opacity = self.params["kappa_abs"](wavelength) +\
                      self.params["cont_weight"]() *\
                      self.params["kappa_cont"](wavelength)
        else:
            opacity = self.params["kappa_abs"](wavelength)
        return opacity

    def _get_radius(self, xx: u.mas, yy: u.mas) -> u.mas:
        """Calculates the radius."""
        if self.radius is None:
            self.radius = np.hypot(xx, yy)
        return self.radius

    def _image_function(self, xx: u.mas, yy: u.mas,
                        wavelength: u.um) -> u.Jy:
        """Calculates a 2D-image from a dust-surface density- and
        temperature profile.

        Parameters
        ----------
        xx : astropy.units.mas
            The x-coordinate grid.
        yy : astropy.units.mas
            The y-coordinate grid.
        wl : astropy.units.um
            Wavelengths.

        Returns
        -------
        image : astropy.units.Jy
        """
        radius, thickness = self._get_radius(xx, yy), 1
        if not np.isinf(self.params["rout"]()):
            radial_profile = np.logical_and(radius > self.params["rin"](),
                                            radius < self.params["rout"]())
        else:
            radial_profile = radius > self.params["rin"]()
        innermost_radius = self.params["rin0"]()\
            if self.params["rin0"] is not None else self.params["rin"]()

        if self.const_temperature:
            temperature = const_temperature(
                radius.value, self.stellar_radius_angular.value,
                self.params["eff_temp"]().value)
        else:
            temperature = temperature_power_law(
                radius.value, innermost_radius.value,
                self.params["inner_temp"]().value, self.params["q"]().value)

        brightness = intensity(
            temperature, wavelength.to(u.cm).value,
            self.params["pixel_size"]().to(u.rad).value)

        if not self.optically_thick:
            surface_density = surface_density_profile(
                radius.value, innermost_radius.value,
                self.params["inner_sigma"]().value, self.params["p"]().value)

            if self.asymmetric_surface_density:
                surface_density *= 1+azimuthal_modulation(
                    xx.value, yy.value, self.params["a"]().value,
                    self.params["phi"]().to(u.rad).value)

            thickness = optical_thickness(surface_density, self._get_opacity(wavelength).value)

        if self.asymmetric_image:
            brightness *= 1+azimuthal_modulation(
                xx.value, yy.value, self.params["a"]().value,
                self.params["phi"]().to(u.rad).value)
        image = radial_profile*brightness*thickness

        image = np.nan_to_num(image, nan=0)
        if OPTIONS["fourier.binning"] is not None:
            image = rebin_image(image, OPTIONS["fourier.binning"])
        if OPTIONS["fourier.padding"] is not None:
            image = pad_image(image, OPTIONS["fourier.padding"])
        return image*u.Jy


class AsymmetricSDTemperatureGradient(TemperatureGradient):
    """A ring defined by a radial temperature profile in r^q
    that is multiplied by an azimuthal modulation.
    and an asymmetric radial dust surface density profile in r^p.

    Parameters
    ----------
    rin : float
        Inner radius of the disk [mas].
    rout : float
        Outer radius of the disk [mas].
    inner_temp : float
        Inner radius temperature [K].
    inner_sigma : float
        Inner surface density [g/cm^2].
    a : float
        Azimuthal modulation amplitude.
    phi : float
        Azimuthal modulation angle [deg].
    q : float
        Power-law exponent for the temperature profile.
    p : float
        Power-law exponent for the dust surface density profile.
    kappa_abs : float
        Dust mass absorption coefficient [cm^2/g].
    dist : float
        Distance of the star [pc].
    pa : float
        Positional angle [deg].
    elong : float
        Elongation of the disk [dimensionless].
    dim : float
        Dimension of the image [px].

    Attributes
    ----------
    params : dict with keys of str and values of Parameter
        Dictionary of parameters.
    """
    name = "Asymmetric Temperature Gradient"
    shortname = "AsymTempGrad"
    asymmetric_surface_density = True


class AsymmetricSDGreyBody(AsymmetricSDTemperatureGradient):
    """A ring defined by a radial temperature profile in r^q
    that is multiplied by an azimuthal modulation.
    and an asymmetric radial dust surface density profile in r^p.

    Parameters
    ----------
    rin : float
        Inner radius of the disk [mas].
    rout : float
        Outer radius of the disk [mas].
    inner_temp : float
        Inner radius temperature [K].
    inner_sigma : float
        Inner surface density [g/cm^2].
    a : float
        Azimuthal modulation amplitude.
    phi : float
        Azimuthal modulation angle [deg].
    q : float
        Power-law exponent for the temperature profile.
    p : float
        Power-law exponent for the dust surface density profile.
    kappa_abs : float
        Dust mass absorption coefficient [cm^2g^-1].
    dist : float
        Distance of the star [pc].
    pa : float
        Positional angle [deg].
    elong : float
        Elongation of the disk [dimensionless].
    dim : float
        Dimension of the image [px].

    Attributes
    ----------
    params : dict with keys of str and values of Parameter
        Dictionary of parameters.
    """
    name = "Asymmetric Grey Body"
    shortname = "AsymGreyBody"
    asymmetric_surface_density = True
    const_temperature = True


class SymmetricSDGreyBodyContinuum(TemperatureGradient):
    """A ring defined by a radial temperature profile in r^q
    that is multiplied by an azimuthal modulation.
    and an asymmetric radial dust surface density profile in r^p.

    Parameters
    ----------
    rin : float
        Inner radius of the disk [mas].
    rout : float
        Outer radius of the disk [mas].
    inner_temp : float
        Inner radius temperature [K].
    inner_sigma : float
        Inner surface density [g/cm^2].
    a : float
        Azimuthal modulation amplitude.
    phi : float
        Azimuthal modulation angle [deg].
    q : float
        Power-law exponent for the temperature profile.
    p : float
        Power-law exponent for the dust surface density profile.
    kappa_abs : float or oimInterp
        Dust mass absorption coefficient [cm2.g-1].
    dist : float
        Distance of the star [pc].
    pa : float
        Positional angle [deg].
    elong : float
        Elongation of the disk.
    dim : float
        Dimension of the image.

    Attributes
    ----------
    params : dict with keys of str and values of Parameter
        Dictionary of parameters.
    """
    name = "Symmetric Continuum Grey Body"
    shortname = "SymContinuumGreyBody"
    const_temperature = True
    continuum_contribution = True


class AsymmetricSDGreyBodyContinuum(AsymmetricSDTemperatureGradient):
    """A ring defined by a radial temperature profile in r^q
    that is multiplied by an azimuthal modulation.
    and an asymmetric radial dust surface density profile in r^p.

    Parameters
    ----------
    rin : float
        Inner radius of the disk [mas].
    rout : float
        Outer radius of the disk [mas].
    inner_temp : float
        Inner radius temperature [K].
    inner_sigma : float
        Inner surface density [g/cm^2].
    a : float
        Azimuthal modulation amplitude.
    phi : float
        Azimuthal modulation angle [deg].
    q : float
        Power-law exponent for the temperature profile.
    p : float
        Power-law exponent for the dust surface density profile.
    kappa_abs : float or oimInterp
        Dust mass absorption coefficient [cm2.g-1].
    dist : float
        Distance of the star [pc].
    pa : float
        Positional angle [deg].
    elong : float
        Elongation of the disk.
    dim : float
        Dimension of the image.

    Attributes
    ----------
    params : dict with keys of str and values of Parameter
        Dictionary of parameters.
    """
    name = "Asymmetric Continuum Grey Body"
    shortname = "AsymContinuumGreyBody"
    asymmetric_surface_density = True
    const_temperature = True
    continuum_contribution = True


# TODO: Check that this is working properly.
def assemble_components(
        parameters: Dict[str, Dict],
        shared_params: Optional[Dict[str, Parameter]] = None
        ) -> List[Component]:
    """Assembles a model from a dictionary of parameters."""
    components = []
    for (component, params) in parameters:
        comp = getattr(sys.modules[__name__], component)
        components.append(comp(**params, **shared_params,
                               **OPTIONS["model.constant_params"]))
    return components
