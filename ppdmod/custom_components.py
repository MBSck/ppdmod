from typing import Optional

import astropy.units as u
import numpy as np

from .component import AnalyticalComponent, NumericalComponent
from .parameter import STANDARD_PARAMETERS, Parameter
from .options import OPTIONS
from .utils import angular_to_distance, calculate_intensity,\
    rebin_image


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

        Notes
        -----
        The formula for the angular diameter $ \delta = \frac{d}{D} $ is used,
        where 'D' is distance to the centre of the star from the observer and
        'd' the distance from the object to the centre of the star.
        This produces an output in radians.
        """
        if self._stellar_angular_radius is None:
            self._stellar_angular_radius = self.params["eff_radius"]().to(u.m)\
                / self.params["dist"]().to(u.m)*u.rad
        return self._stellar_angular_radius

    def _image_function(self, xx: u.mas, yy: u.mas,
                        wavelength: Optional[u.m] = None,
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
        return self._image*calculate_intensity(self.params["eff_temp"](),
                                               wavelength,
                                               self.stellar_radius_angular)

    def _visibility_function(self,
                             wavelength: Optional[u.um] = None) -> np.ndarray:
        """The component's _visibility_function."""
        if self._visibility is None:
            self._visibility = np.ones((self.params["dim"](),
                                        self.params["dim"]()))
        return self._visibility\
            * calculate_intensity(self.params["eff_temp"](),
                                  wavelength,
                                  self.stellar_radius_angular).value


# TODO: Think of doing all conversions in properties -> Quicker?
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
    asymmetric = False
    asymmetric_image = False
    asymmetric_surface_density = False
    optically_thick = False
    const_temperature = False
    continuum_contribution = False

    def __init__(self, **kwargs):
        """The class's constructor."""
        super().__init__(**kwargs)
        self.params["dist"] = Parameter(**STANDARD_PARAMETERS["dist"])
        self.params["eff_temp"] = Parameter(**STANDARD_PARAMETERS["eff_temp"])
        self.params["eff_radius"] = Parameter(**STANDARD_PARAMETERS["eff_radius"])

        self.params["rin"] = Parameter(**STANDARD_PARAMETERS["rin"])
        self.params["rout"] = Parameter(**STANDARD_PARAMETERS["rout"])

        if self.asymmetric:
            self.params["a"] = Parameter(**STANDARD_PARAMETERS["a"])
            self.params["phi"] = Parameter(**STANDARD_PARAMETERS["phi"])

        self.params["q"] = Parameter(**STANDARD_PARAMETERS["q"])
        self.params["p"] = Parameter(**STANDARD_PARAMETERS["p"])
        self.params["inner_temp"] = Parameter(**STANDARD_PARAMETERS["inner_temp"])

        if self.const_temperature:
            self.params["q"].free = False
            self.params["inner_temp"].free = False

        self.params["inner_sigma"] = Parameter(**STANDARD_PARAMETERS["inner_sigma"])
        self.params["kappa_abs"] = Parameter(**STANDARD_PARAMETERS["kappa_abs"])
        if self.continuum_contribution:
            self.params["cont_weight"] = Parameter(**STANDARD_PARAMETERS["cont_weight"])
            self.params["kappa_cont"] = Parameter(**STANDARD_PARAMETERS["kappa_cont"])
        self._eval(**kwargs)

    def _calculate_azimuthal_modulation(self, xx: u.mas, yy: u.mas) -> np.ndarray:
        r"""Calculates the azimuthal modulation.

        Parameters
        ----------
        xx : astropy.units.mas
            The x-coordinate grid.
        yy : astropy.units.mas
            The y-coordinate grid.

        Returns
        -------
        azimuthal_modulation : astropy.units.one

        Notes
        -----
        Derived via trigonometry from Lazareff et al. 2017's:

        $ F(r) = F_{0}(r)\cdot\left(1+\sum_{j=1}^{m}()c_{j}\cos(j\phi)+s_{j}\sin(j\phi)\right)$
        """
        return self.params["a"]()\
            * np.cos(np.arctan2(yy, xx)-self.params["phi"]().to(u.rad))

    def _calculate_surface_density_profile(self, radius: u.mas,
                                           xx: u.mas, yy: u.mas) -> u.g/u.cm**2:
        """Calculates the surface density profile.

        This can be azimuthally varied if so specified.

        Parameters
        ----------
        xx : astropy.units.mas
            The x-coordinate grid.
        yy : astropy.units.mas
            The y-coordinate grid.

        Returns
        -------
        surface_density_profile : astropy.units.g/astropy.units.cm**2

        Notes
        -----
        """
        surface_density = self.params["inner_sigma"]()\
            * (radius / self.params["rin"]())**(-self.params["p"]())
        if self.asymmetric_surface_density:
            return surface_density\
                * (1+self._calculate_azimuthal_modulation(xx, yy))
        return surface_density

    def _calculate_optical_depth(self, radius: u.mas,
                                 xx: u.mas, yy: u.mas,
                                 wavelength: u.um) -> u.one:
        """Calculates and returns the optical depth

        Parameters
        ----------
        xx : astropy.units.mas
            The x-coordinate grid.
        yy : astropy.units.mas
            The y-coordinate grid.
        wavelength : astropy.units.um

        Returns
        -------
        optical_depth : u.one
        """
        sigma_profile = self._calculate_surface_density_profile(radius, xx, yy)
        if self.continuum_contribution:
            opacity = self.params["kappa_abs"](wavelength) +\
                    self.params["cont_weight"]() *\
                    self.params["kappa_cont"](wavelength)
        else:
            opacity = self.params["kappa_abs"](wavelength)
        return -sigma_profile*opacity

    def _calculate_temperature_profile(self, radius: u.mas) -> u.K:
        """Calculates the temperature profile.

        Can be specified to be either as a r^q power law or an a
        constant/idealised temperature profile derived from the star's
        temperature, its radius and the observer's distance to the star.
        It will then be only contingent on those values.

        Parameters
        ----------
        radius : astropy.units.mas

        Returns
        -------
        temperature_profile : astropy.units.K

        Notes
        -----
        In case of a radial power law the formula is

        .. math:: T = T_0 * (1+\\frac{r}{R_0})^\\q.

        In case of a constant grey body profile the stellar radius is
        calculated from its lumionsity via

        .. math:: R_* = \\sqrt{\\frac{L_*}{4\\pi\\sigma_sb\\T_*^4}}.

        And with this the individual grain's temperature profile is

        .. math:: T_{grain} = \\sqrt{\\frac{R_*}{2r}}\\cdot T_*.
        """
        if self.const_temperature:
            radius = angular_to_distance(radius, self.params["dist"]())
            return np.sqrt(self.params["eff_radius"]().to(u.m)/(2*radius))\
                * self.params["eff_temp"]()
        return self.params["inner_temp"]()\
            * (radius / self.params["rin"]())**(-self.params["q"]())

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
        # TODO: Is there a difference between first setting all 0
        # and then calculating or the other way around?
        radius = np.hypot(xx, yy)
        if not np.isinf(self.params["rout"]()):
            radial_profile = np.logical_and(radius > self.params["rin"](),
                                            radius < self.params["rout"]())
        else:
            radial_profile = radius > self.params["rin"]()
        temperature_profile = self._calculate_temperature_profile(radius)
        spectral_density = calculate_intensity(temperature_profile,
                                               wavelength,
                                               self.params["pixel_size"]())
        # TODO: Test if this is all correct.
        image = radial_profile*spectral_density
        if not self.optically_thick:
            image *= (1-np.exp(self._calculate_optical_depth(radius, xx,
                                                             yy, wavelength)))
        image = np.nan_to_num(image, nan=0)
        if self.asymmetric_image:
            image = self._calculate_azimuthal_modulation(xx, yy, wavelength)\
                * (1+image)

        if OPTIONS["fourier.binning"] is not None:
            image = rebin_image(image, OPTIONS["fourier.binning"])
        return image


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
    asymmetric = True
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
    asymmetric = True
    asymmetric_surface_density = True
    const_temperature = True


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
    asymmetric = True
    asymmetric_surface_density = True
    const_temperature = True
    continuum_contribution = True
