from typing import Optional, Tuple

import astropy.units as u
import numpy as np
from astropy.modeling.models import BlackBody
from scipy.special import j0, jvn

from ._spectral_cy import grid
from .fft import compute_real2Dfourier_transform
from .parameter import STANDARD_PARAMETERS, Parameter
from .options import OPTIONS
from .utils import rebin_image, upbin_image, get_new_dimension,\
        distance_to_angular


class Component:
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
    name = "Generic component"
    shortname = "GenComp"
    description = "This is the class from which all components are derived."
    _elliptic = False

    def __init__(self, **kwargs):
        """The class's constructor."""
        self.params = {}
        self.params["x"] = Parameter(**STANDARD_PARAMETERS["x"])
        self.params["y"] = Parameter(**STANDARD_PARAMETERS["y"])
        self.params["dim"] = Parameter(**STANDARD_PARAMETERS["dim"])
        self.params["pixel_size"] = Parameter(
            **STANDARD_PARAMETERS["pixel_size"])
        self.params["pa"] = Parameter(**STANDARD_PARAMETERS["pa"])
        self.params["elong"] = Parameter(**STANDARD_PARAMETERS["elong"])

        if not self.elliptic:
            self.params["pa"].free = False
            self.params["elong"].free = False
        self._eval(**kwargs)

    @property
    def elliptic(self) -> bool:
        """Gets if the component is elliptic."""
        return self._elliptic

    @elliptic.setter
    def elliptic(self, value: bool) -> None:
        """Sets the position angle and the parameters to free or false
        if elliptic is set."""
        if value:
            self.params["pa"].free = True
            self.params["elong"].free = True
        else:
            self.params["pa"].free = False
            self.params["elong"].free = False
        self._elliptic = value

    def _eval(self, **kwargs):
        """Sets the parameters (values) from the keyword arguments."""
        for key, value in kwargs.items():
            if key in self.params:
                if isinstance(value, Parameter):
                    self.params[key] = value
                else:
                    self.params[key].value = value

    def _calculate_internal_grid(
            self, dim: int, pixel_size: u.mas
            ) -> Tuple[u.Quantity[u.mas], u.Quantity[u.mas]]:
        """Calculates the model grid.

        Parameters
        ----------
        dim : float, optional
        pixel_size : float, optional

        Returns
        -------
        xx : astropy.units.mas
            The x-coordinate grid.
        yy : astropy.units.mas
            The y-coordinate grid.
        """
        elong, pa = self.params["elong"](), self.params["pa"]()
        elong = elong.value if elong is not None else elong
        pa = pa.value if pa is not None else pa
        return grid(dim, pixel_size.value, elong, pa, self.elliptic)

    def _translate_fourier_transform(self, ucoord: u.m, vcoord: u.m,
                                     wavelength: u.um) -> u.one:
        """Translate the coordinates of the fourier transform."""
        x, y = map(lambda x: self.params[x]().to(u.rad), ["x", "y"])
        ucoord, vcoord = map(
            lambda x: (u.Quantity(value=x, unit=u.m)/wavelength.to(u.m))/u.rad,
            [ucoord, vcoord])
        return np.exp(-2*1j*np.pi*(ucoord*x+vcoord*y)).value

    def _translate_coordinates(
            self, xx: u.mas, yy: u.mas
            ) -> Tuple[u.Quantity[u.mas], u.Quantity[u.mas]]:
        """Shifts the coordinates according to an offset."""
        xx, yy = map(lambda x: u.Quantity(value=x, unit=u.mas), [xx, yy])
        return xx-self.params["x"](), yy-self.params["y"]()


class AnalyticalComponent(Component):
    """Class for all analytically calculated components."""
    name = "Analytical Component"
    shortname = "AnaComp"
    description = "This is the class from which all"\
                  "analytical components are derived."

    def _image_function(self, xx: u.mas, yy: u.mas,
                        wavelength: Optional[u.Quantity[u.um]] = None
                        ) -> Optional[u.Quantity]:
        """Calculates the image from a 2D grid.

        Parameters
        ----------
        xx : astropy.units.mas
            The x-coordinate grid.
        yy : astropy.units.mas
            The y-coordinate grid.
        wavelength : u.m, optional

        Returns
        -------
        image : astropy.units.Quantity, Optional
        """
        return

    def _visibility_function(self, dim: int, pixel_size: u.mas,
                             wavelength: Optional[u.Quantity[u.um]] = None
                             ) -> np.ndarray:
        """Calculates the complex visibility of the the component's image.

        Parameters
        ----------
        wavelength : astropy.units.um, optional

        Returns
        -------
        complex_visibility_function : numpy.ndarray
        """
        return

    def calculate_image(self, dim: Optional[float] = None,
                        pixel_size: Optional[float] = None,
                        wavelength: Optional[u.Quantity[u.um]] = None) -> u.Jy:
        """Calculates a 2D image.

        Parameters
        ----------
        dim : float
            The dimension [px].
        pixel_size : float
            The size of a pixel [mas].
        wavelength : astropy.units.um, optional

        Returns
        -------
        image : astropy.units.Quantity
        """
        dim = self.params["dim"]() if dim is None else dim
        pixel_size = self.params["pixel_size"]()\
            if pixel_size is None else pixel_size
        dim = u.Quantity(value=dim, unit=u.one, dtype=int)
        pixel_size = u.Quantity(value=pixel_size, unit=u.mas)
        dim = get_new_dimension(
                dim, OPTIONS["fourier.binning"], OPTIONS["fourier.padding"])
        x_arr, y_arr = self._calculate_internal_grid(dim, pixel_size)
        return self._image_function(x_arr, y_arr, wavelength)

    def calculate_complex_visibility(
            self, dim: Optional[float] = None,
            pixel_size: Optional[float] = None,
            wavelength: Optional[u.Quantity[u.um]] = None) -> np.ndarray:
        """Calculates the complex visibility of the the component's image.

        Parameters
        ----------
        wavelength : astropy.units.um, optional

        Returns
        -------
        complex_visibility_function : numpy.ndarray
        """
        dim = self.params["dim"]() if dim is None else dim
        pixel_size = self.params["pixel_size"]()\
            if pixel_size is None else pixel_size
        dim = u.Quantity(value=dim, unit=u.one, dtype=int)
        pixel_size = u.Quantity(value=pixel_size, unit=u.mas)
        dim = get_new_dimension(
                dim, OPTIONS["fourier.binning"], OPTIONS["fourier.padding"])
        return self._visibility_function(dim, pixel_size, wavelength)


class HankelComponent(Component):
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
    continuum_contribution = False

    def __init__(self, **kwargs):
        """The class's constructor."""
        super().__init__(**kwargs)
        self._stellar_angular_radius = None

        self.params["dist"] = Parameter(**STANDARD_PARAMETERS["dist"])
        self.params["eff_temp"] = Parameter(**STANDARD_PARAMETERS["eff_temp"])
        self.params["eff_radius"] = Parameter(**STANDARD_PARAMETERS["eff_radius"])

        self.params["rin0"] = Parameter(**STANDARD_PARAMETERS["rin0"])

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

    def _calculate_internal_grid(self, dim: int) -> u.mas:
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
        if OPTIONS["model.gridtype"] == "linear":
            return np.linspace(rin.value, rout.value, dim)*self.params["rin"].unit
        return np.logspace(np.log10(rin.value),
                           np.log10(rout.value), dim)*self.params["rin"].unit

    def _get_opacity(self, wavelength: u.um) -> u.cm**2/u.g:
        """Set the opacity from wavelength."""
        if self.continuum_contribution:
            opacity = self.params["kappa_abs"](wavelength) +\
                      self.params["cont_weight"]() *\
                      self.params["kappa_cont"](wavelength)
        else:
            opacity = self.params["kappa_abs"](wavelength)
        return opacity

    def _brightness_profile_function(self, wavelength: u.um) -> u.Jy:
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
        radius = self._calculate_internal_grid(self.params["dim"]())
        thickness_profile = 1

        # TODO: Think of a way how to implement the innermost radius
        # and at the same time keep the grid as it is.
        innermost_radius = self.params["rin0"]()\
            if self.params["rin0"]() != 0 else self.params["rin"]()

        if self.const_temperature:
            temp_profile = np.sqrt(self.stellar_radius_angular/(2.0*radius))\
                    * self.params["eff_temp"]()
        else:
            temp_profile = self.params["inner_temp"]()\
                    * (radius/innermost_radius)**(-self.params["q"]())

        brightness_profile = BlackBody(temp_profile)(wavelength)

        if not self.optically_thick:
            surface_density_profile = self.params["inner_sigma"]()\
                    * (radius/innermost_radius)**(-self.params["p"]())

            # if self.asymmetric_surface_density:
            #     surface_density *= 1+azimuthal_modulation(
            #         xx, yy, self.params["a"]().value,
            #         self.params["phi"]().to(u.rad).value)
            thickness_profile = (1-np.exp(-surface_density_profile\
                    * self._get_opacity(wavelength)))

        return brightness_profile*thickness_profile

    def get_total_flux(self, brightness_profile: u.erg/(u.rad**2*u.s*u.Hz),
                       radius: u.mas) -> u.Jy:
        """Calculates the total flux from the hankel transformation."""
        radius = radius.to(u.rad)
        return 2.*np.pi*np.trapz(radius*brightness_profile, radius).to(u.Jy)\
                * self.params["pa"]().to(u.rad).value

    def hankel_transform(self, brightness_profile: u.erg/(u.rad**2*u.s*u.Hz), radius: u.mas,
                         ucoord: u.m, vcoord: u.m, wavelength: u.um) -> np.ndarray:
        """Calculates the hankel transformation for a modulated ring."""
        # pad = 1 if OPTIONS['FTPaddingFactor'] is None else OPTIONS['FTPaddingFactor']
        # fov = radius[-1]-radius[0]
        # dsfreq0 = 1/(fov*pad).value
        # sfreq0 = np.linspace(0, pad*nr-1, pad*nr)*dsfreq0
        radius = radius.to(u.rad)
        baselines = (np.hypot(ucoord, vcoord)/wavelength.to(u.m))*u.rad
        baseline_angles = np.arctan2(vcoord, ucoord)

        visibilities = []
        for baseline, baseline_angle in zip(baselines, baseline_angle):
            visibility = self.params["pa"]().to(u.rad).value*2*np.pi*np.trapz(
                    radius*brightness_profile*j0(2.*np.pi*radius.value*baseline.value), radius)
            modulations = []
            for order in OPTIONS["model.modulation.order"]:
                modulation = (-1j)**order*self.params["a"]()*np.cos(baseline_angle-self.params["phi"]().to(u.rad))\
                        * np.trapz(radius*brightness_profile*jv(order, 2.*np.pi*radius.value*baseline.value), radius)
                modulations.append(modulation)
            modulations = np.sum(modulations).to(u.Jy)
            visibilities.append(visibility.to(u.Jy)+modulations)
        return visibilities

    def calculate_complex_visibility(
            self, ucoord: u.m, vcoord: u.m,
            wavelength: Optional[u.Quantity[u.um]] = None) -> np.ndarray:
        """Calculates the complex visibility of the the component's image.

        Parameters
        ----------
        ucoord : astropy.units.m
        vcoord : astropy.units.m
        wavelength : astropy.units.um, optional

        Returns
        -------
        complex_visibility_function : numpy.ndarray
        """
        radius = self._calculate_internal_grid(self.params["dim"]())
        brightness_profile = self._brightness_profile_function(wavelength)
        return self.hankel_transform(brightness_profile, radius, ucoord, vcoord, wavelength)


class NumericalComponent(Component):
    """Base class with increased computational performance for numerical
    calculations.

    Parameters
    ----------
    pixel_size : float
        The size of a pixel [mas].
    pa : float
        Positional angle [deg].
    elong : float
        Elongation of the disk [dimensionless].
    dim : float
        The dimension [px].

    Attributes
    ----------

    Notes
    -----
    This class will automaticall set-up a cache directory in order to speed up
    the calculations. The directory itself can be set via the 'cache_dir' keyword.
    """
    elliptic = False

    def _image_function(self, xx: u.mas,
                        yy: u.mas, wavelength: u.um) -> u.Jy:
        """Calculates the image from a 2D grid.

        Parameters
        ----------
        xx : astropy.units.mas
            The x-coordinate grid.
        yy : astropy.units.mas
            The y-coordinate grid.
        wavelength : astropy.units.um, optional

        Returns
        -------
        image : astropy.units.Quantity, optional
        """
        return

    def calculate_image(self, dim: Optional[float] = None,
                        pixel_size: Optional[float] = None,
                        wavelength: Optional[u.Quantity[u.um]] = None) -> u.Jy:
        """Calculates a 2D image.

        Parameters
        ----------
        dim : float
            The dimension [px].
        pixel_size : float
            The size of a pixel [mas].
        wavelength : astropy.units.um, optional

        Returns
        -------
        image : astropy.units.Quantity
        """
        dim = self.params["dim"]() if dim is None else dim
        pixel_size = self.params["pixel_size"]()\
            if pixel_size is None else pixel_size
        dim = u.Quantity(value=dim, unit=u.one, dtype=int)
        pixel_size = u.Quantity(value=pixel_size, unit=u.mas)

        if OPTIONS["model.matryoshka"]:
            image = None
            new_dim = get_new_dimension(
                    dim, OPTIONS["model.matryoshka.binning_factors"][0])
            for binning_factor in OPTIONS["model.matryoshka.binning_factors"]:
                binning_factor = binning_factor if binning_factor is not None else 0
                if image is None:
                    x_arr, y_arr = self._calculate_internal_grid(
                            new_dim, pixel_size*2**binning_factor)
                    image_part = self._image_function(x_arr, y_arr, wavelength)
                    image = upbin_image(image_part, binning_factor)\
                            * (2**binning_factor*2**binning_factor)
                else:
                    x_arr, y_arr = self._calculate_internal_grid(
                            new_dim, pixel_size*2**-binning_factor)
                    image_part = self._image_function(x_arr, y_arr, wavelength)
                    image_part = rebin_image(image_part, binning_factor)
                    start = (image.shape[0]-image_part.shape[0])//2
                    end = start + image_part.shape[0]
                    image[start:end, start:end] =\
                            image_part/(2**binning_factor*2**binning_factor)
        else:
            x_arr, y_arr = self._calculate_internal_grid(dim, pixel_size)
            image = self._image_function(x_arr, y_arr, wavelength)
        if OPTIONS["fourier.binning"] is not None:
            image = rebin_image(image, OPTIONS["fourier.binning"])
        if OPTIONS["fourier.padding"] is not None:
            image = pad_image(image, OPTIONS["fourier.padding"])
        return image

    def calculate_complex_visibility(
            self, dim: Optional[float] = None,
            pixel_size: Optional[float] = None,
            wavelength: Optional[u.Quantity[u.um]] = None) -> np.ndarray:
        """Calculates the complex visibility of the the component's image.

        Parameters
        ----------
        wavelength : astropy.units.um, optional

        Returns
        -------
        complex_visibility_function : numpy.ndarray
        """
        dim = self.params["dim"]() if dim is None else dim
        pixel_size = self.params["pixel_size"]()\
            if pixel_size is None else pixel_size
        dim = u.Quantity(value=dim, unit=u.one, dtype=int)
        pixel_size = u.Quantity(value=pixel_size, unit=u.mas)
        image = self.calculate_image(dim, pixel_size, wavelength)
        return compute_real2Dfourier_transform(image.value)
