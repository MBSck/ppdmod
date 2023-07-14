import astropy.units as u
import astropy.constants as const
import numpy as np
from astropy.modeling import models

from component import AnalyticalComponent, NumericalComponent
from parameter import Parameter
from utils import convert_radial_profile_to_meter, calculate_intensity


class Star(AnalyticalComponent):
    """Star defined as an analytical component.

    Parameters
    ----------
    x : int
        x pos of the component (in mas). The default is 0.
    y : int
        y pos of the component (in mas). The default is 0.
    eff_temp : float
        The star's temperature.
    dist : float
        Distance to the star.
    lum : float
        The star's luminosity.
    """
    name = "Star"
    shortname = "St"
    description = "The flux of a star."

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._stellar_radius = None
        self._stellar_angular_radius = None
        self._image = None

        self.params["eff_temp"] = Parameter(name="eff_temp", value=0,
                                            unit=u.K, free=False,
                                            description="The star's temperature")
        self.params["dist"] = Parameter(name="dist",
                                        value=0,
                                        unit=u.pc, free=False,
                                        description="Distance to the star")
        self.params["lum"] = Parameter(name="lum",
                                       value=0,
                                       unit=u.Lsun, free=False,
                                       description="The star's luminosity")

        self._eval(**kwargs)

    @property
    def stellar_radius(self) -> u.m:
        """Calculates the stellar radius.

        Returns
        -------
        stellar_radius : astropy.units.m
            The star's radius.
        """
        if self._stellar_radius is None:
            luminosity = (self.params["lum"].value*self.params["lum"].unit).to(u.W)
            self._stellar_radius = np.sqrt(luminosity /
                                           (4*np.pi*const.sigma_sb*(
                                            self.params["eff_temp"].value
                                            * self.params["eff_temp"].unit)**4))
        return self._stellar_radius

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
        The formula for the angular diameter $ \delta = \frac{d}{D} $ is used.
        This produces an output in radians.
        """
        if self._stellar_angular_radius is None:
            distance = self.params["dist"].value*self.params["dist"].unit
            self._stellar_angular_radius = (self.stellar_radius.to(u.m) /
                                            distance.to(u.m)*u.rad).to(u.mas)
        return self._stellar_angular_radius

    def _calculate_flux(self, wl: u.m) -> np.ndarray:
        """Calculates a temperature and wavelength dependent blackbody
        distribution via Planck's law.

        Parameters
        ----------
        wl : astropy.units.m
            Wavelengths [m].

        Returns
        -------
        blackbody_distribution : astropy.units.Jy
            The star's flux.
        """
        plancks_law = models.BlackBody(
            temperature=self.params["eff_temp"].value*self.params["eff_temp"].unit)
        spectral_radiance = plancks_law(wl*u.m).to(
            u.erg/(u.cm**2*u.Hz*u.s*u.mas**2))
        return (np.pi*spectral_radiance *
                self.stellar_radius_angular**2).to(u.Jy).value

    def _image_function(self, xx, yy, wl):
        if self._image is None:
            image = xx*0
            val = np.abs(xx)+np.abs(yy)
            idx = np.unravel_index(np.argmin(val), np.shape(val))
            image[idx] = 1
        return self._image*self._calculate_flux(wl)

    def _visibility_function(self, ucoord, vcoord, rho, wl):
        return self._calculate_flux(wl)


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
    Tin : float
        Inner radius temperature [K].
    Mdust : float
        Mass of the dusty disk [M_sun].
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
    pixSize : float
        The pixel size of the image [mas/px].
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
    pixSize : float
        Pixel size [mas].
    _wl : numpy.ndarray
        Array of wavelength values [micron].
    """
    name = "Asymmetric Temperature Gradient"
    shortname = "AsymTempGrad"
    elliptic = True
    asymmetric = True
    asymmetric_image = True
    asymmetric_surface_density = False
    const_temperature = False
    continuum_contribution = False

    def __init__(self, **kwargs):
        """The class's constructor."""
        super().__init__(**kwargs)
        self.normalizeImage = False
        self.params["rin"] = Parameter(name="rin", value=0, unit=u.mas,
                                       description="Inner radius of the disk")
        self.params["rout"] = Parameter(name="rout", value=0, unit=u.mas,
                                        description="Outer radius of the disk")

        if self.asymmetric:
            self.params["a"] = Parameter(name="a", value=0, unit=u.one,
                                         description="Azimuthal modulation amplitude")
            self.params["phi"] = Parameter(name="phi", value=0, unit=u.deg,
                                           description="Azimuthal modulation angle")

        self.params["q"] = Parameter(name="q", value=0, unit=u.one,
                                     description="Power-law exponent for the temperature profile")
        self.params["p"] = Parameter(name="p", value=0, unit=u.one,
                                     description="Power-law exponent for the surface density profile")

        if self.const_temperature:
            self.params["q"].free = False

        self.params["p"].description = "Power-law exponent for the dust surface density profile"
        self.params["Mdust"] = Parameter(name="Mdust", value=0, unit=u.M_sun,
                                         description="Mass of the dusty disk")
        self.params["Tin"] = Parameter(name="Tin", value=0,
                                       unit=u.K, free=False,
                                       description="Inner radius temperature")
        self.params["kappa_abs"] = Parameter(name="kappa_abs", value=0,
                                             unit=u.cm**2/u.g, free=False,
                                             description="Dust mass absorption coefficient")
        self.params["dist"] = Parameter(name="dist", value=0,
                                        unit=u.pc, free=False,
                                        description="Distance of the star")
        self.params["Teff"] = Parameter(name="Teff", value=0,
                                        unit=u.K, free=False,
                                        description="The star's effective Temperature")
        self.params["lum"] = Parameter(name="lum", value=0,
                                       unit=u.Lsun, free=False,
                                       description="The star's luminosity")

        if self.continuum_contribution:
            self.params["cont_weight"] = Parameter(name="cont_weight", value=0,
                                                   unit=u.one, free=True,
                                                   description="Dust mass continuum absorption coefficient's weight")
            self.params["kappa_cont"] = Parameter(name="kappa_cont", value=0,
                                                  unit=u.cm**2/u.g, free=False,
                                                  description="Continuum dust mass absorption coefficient")

        self._wl = None  # None value <=> All wavelengths (from Data)
        self._eval(**kwargs)

    def _azimuthal_modulation(self, xx: np.ndarray,
                              yy: np.ndarray, wl: np.ndarray) -> np.ndarray:
        """Calculates the azimuthal modulation.

        Parameters
        ----------
        xx : numpy.ndarray
            The x-coordinate grid
        yy : numpy.ndarray
            The y-coordinate grid
        wl : numpy.ndarray
            Wavelengths.

        Returns
        -------
        azimuthal_modulation : numpy.ndarray
        """
        phi = self.params["phi"](wl)*self.params["phi"].unit.to(u.rad)
        return self.params["a"](wl)*np.cos(np.arctan2(yy, xx)-phi)

    def _surface_density_profile(self, xx, yy, wl):
        """Calculates the surface density profile.

        This can be azimuthally varied if so specified.

        Parameters
        ----------
        xx : numpy.ndarray
            The x-coordinate grid [mas].
        yy : numpy.ndarray
            The y-coordinate grid [mas].
        wl : numpy.ndarray
            Wavelengths [micron].

        Returns
        -------
        surface_density_profile : np.ndarray
            The surface density profile [g/cm^2].
        """
        dist = self.params["dist"](wl)
        rin, rout = map(lambda x: self.params[x](wl), ["rin", "rout"])
        rin_cm = convert_radial_profile_to_meter(rin, dist).to(u.cm).value
        rout_cm = convert_radial_profile_to_meter(rout, dist).to(u.cm).value

        p = self.params["p"](wl)
        dust_mass = self.params["Mdust"](wl)*const.M_sun.value*1e3
        if p == 2:
            sigma_in = dust_mass/(2.*np.pi*np.log(rout_cm/rin_cm)*rin_cm**2)
        else:
            f = ((rout_cm/rin_cm)**(2-p)-1)/(2-p)
            sigma_in = dust_mass/(2.*np.pi*f*rin_cm**2)

        sigma_profile = sigma_in*(np.sqrt(xx**2+yy**2) / rin)**(-p)
        if self.asymmetric_surface_density:
            return sigma_profile*(1+self._azimuthal_modulation(xx, yy, wl))
        return sigma_profile

    def _optical_depth(self, xx, yy, wl):
        """Calculates and returns the optical depth

        Parameters
        ----------
        xx : numpy.ndarray
            The x-coordinate grid [mas].
        yy : numpy.ndarray
            The y-coordinate grid [mas].
        wl : numpy.ndarray
            Wavelengths [micron].

        Returns
        -------
        optical_depth : np.ndarray
            The optical depth.
        """
        sigma_profile = self._surface_density_profile(xx, yy, wl)
        if self.continuum_contribution:
            opacities = self.params["kappa_abs"](wl) +\
                        self.params["cont_weight"](wl) *\
                        self.params["kappa_cont"](wl)
        else:
            opacities = self.params["kappa_abs"](wl)

        optical_depth = []
        for opacity in opacities:
            optical_depth.append(-sigma_profile*opacity)
        return np.array(optical_depth)

    def _temperature_profile(self, r, wl):
        """Calculates the temperature profile.

        Can be specified to be either as a r^q power law or an a
        constant/idealised temperature profile derived from the star's
        luminosity and the observer's distance to the star and contingent
        only on those values.

        Parameters
        ----------
        r : numpy.ndarray
            Radial grid [mas].
        wl : numpy.ndarray
            Wavelengths [micron].

        Returns
        -------
        temperature_profile : numpy.ndarray
            The temperature profile [K].

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
            radius = convert_radial_profile_to_meter(r, self.params["dist"](wl))
            luminosity = (self.params["lum"](wl) *
                          self.params["lum"].unit).to(u.W)
            stellar_temperature = self.params["Teff"](wl)*self.params["Teff"].unit
            stellar_radius = np.sqrt(
                luminosity/(4*np.pi*const.sigma_sb*stellar_temperature**4))
            return (np.sqrt(stellar_radius/(2*radius))*stellar_temperature).value
        q, inner_temp = map(lambda x: self.params[x](wl), ["q", "Tin"])
        return inner_temp*(r / self.params["rin"](wl))**(-q)

    def _image(self, xx: np.ndarray,
               yy: np.ndarray, wl: np.ndarray) -> np.ndarray:
        """Combines the various radial profiles into an image.

        If physical output is specified, the model will produce Jansky per
        pixel else unitless intensity.

        Parameters
        ----------
        xx : numpy.ndarray
            The x-coordinate grid [mas].
        yy : numpy.ndarray
            The y-coordinate grid [mas].
        wl : numpy.ndarray
            Wavelengths [micron].

        Returns
        -------
        image : numpy.ndarray
        """
        r = np.sqrt(xx**2+yy**2)
        rin, rout = map(lambda x: self.params[x](wl), ["rin", "rout"])
        radial_profile = np.logical_and(r > rin, r < rout).astype(int)
        temperature_profile = self._temperature_profile(r, wl)
        spectral_density = calculate_intensity(wl, temperature_profile, self.pixSize)
        return np.nan_to_num(radial_profile * spectral_density *
                             (1 - np.exp(self._optical_depth(xx, yy, wl))), nan=0)

    def _imageFunction(self, xx: np.ndarray,
                       yy: np.ndarray, wl: np.ndarray) -> np.ndarray:
        """Calculates a 2D-image from a dust-surface density- and
        temperature profile.

        Parameters
        ----------
        xx : numpy.ndarray
            The x-coordinate grid
        yy : numpy.ndarray
            The y-coordinate grid
        wl : numpy.ndarray
            Wavelengths.

        Returns
        -------
        image : numpy.ndarray
        """
        if self.asymmetric_image:
            img = self._image(xx, yy, wl) * \
                (1+self._azimuthal_modulation(xx, yy, wl))
        else:
            img = self._image(xx, yy, wl)
        return img


class AsymmetricSDTemperatureGradient(NumericalComponent):
    """A ring defined by a radial temperature profile in r^q
    that is multiplied by an azimuthal modulation.
    and an asymmetric radial dust surface density profile in r^p.

    Parameters
    ----------
    rin : float
        Inner radius of the disk [mas].
    rout : float
        Outer radius of the disk [mas].
    Tin : float
        Inner radius temperature [K].
    Mdust : float
        Mass of the dusty disk [M_sun].
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
    pixSize : float
        Pixel size [mas].
    _wl : numpy.ndarray
        Array of wavelength values [micron].
    """
    name = "Asymmetric Temperature Gradient"
    shortname = "AsymTempGrad"
    elliptic = True
    asymmetric = True
    asymmetric_image = False
    asymmetric_surface_density = True
    const_temperature = False
    continuum_contribution = False


class AsymmetricSDGreyBody(NumericalComponent):
    """A ring defined by a radial temperature profile in r^q
    that is multiplied by an azimuthal modulation.
    and an asymmetric radial dust surface density profile in r^p.

    Parameters
    ----------
    rin : float
        Inner radius of the disk [mas].
    rout : float
        Outer radius of the disk [mas].
    Tin : float
        Inner radius temperature [K].
    Mdust : float
        Mass of the dusty disk [M_sun].
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
    pixSize : float
        Pixel size [mas].
    _wl : numpy.ndarray
        Array of wavelength values [micron].
    """
    name = "Asymmetric Temperature Gradient"
    shortname = "AsymTempGrad"
    elliptic = True
    asymmetric = True
    asymmetric_image = False
    asymmetric_surface_density = True
    const_temperature = True
    continuum_contribution = False


class AsymmetricSDGreyBodyContinuum(NumericalComponent):
    """A ring defined by a radial temperature profile in r^q
    that is multiplied by an azimuthal modulation.
    and an asymmetric radial dust surface density profile in r^p.

    Parameters
    ----------
    rin : float
        Inner radius of the disk [mas].
    rout : float
        Outer radius of the disk [mas].
    Tin : float
        Inner radius temperature [K].
    Mdust : float
        Mass of the dusty disk [M_sun].
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
    pixSize : float
        Pixel size [mas].
    _wl : numpy.ndarray
        Array of wavelength values [micron].
    """
    name = "Asymmetric Temperature Gradient"
    shortname = "AsymTempGrad"
    elliptic = True
    asymmetric = True
    asymmetric_image = False
    asymmetric_surface_density = True
    const_temperature = True
    continuum_contribution = True


if __name__ == "__main__":
    test = Star(eff_temp=7800, dist=150, lum=19)
    breakpoint()
