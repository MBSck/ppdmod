from typing import Optional, Tuple
from warnings import warn

import astropy.units as u
import numpy as np

from ._spectral_cy import grid
from .fft import compute_real2Dfourier_transform
from .parameter import STANDARD_PARAMETERS, Parameter
from .options import OPTIONS
from .utils import rebin_image, upbin_image, get_new_dimension


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
