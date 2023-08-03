from typing import Optional, Tuple

import astropy.units as u
import numpy as np

from .fft import compute_2Dfourier_transform
from .options import OPTIONS
from .parameter import STANDARD_PARAMETERS, Parameter
from .utils import get_binned_dimension


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

    def __init__(self, **kwargs):
        """The class's constructor."""
        self.params = {}
        self.params["x"] = Parameter(**STANDARD_PARAMETERS["x"])
        self.params["y"] = Parameter(**STANDARD_PARAMETERS["y"])
        self.params["dim"] = Parameter(**STANDARD_PARAMETERS["dim"])
        self._eval(**kwargs)

    def _eval(self, **kwargs):
        """Sets the parameters (values) from the keyword arguments."""
        for key, value in kwargs.items():
            if key in self.params:
                if isinstance(value, Parameter):
                    self.params[key] = value
                else:
                    self.params[key].value = value

    def _translate_fourier_transform(self, ucoord, vcoord):
        x = self.params["x"]().to(u.rad)
        y = self.params["y"]().to(u.rad)
        return np.exp(-2*1j*np.pi*(ucoord*x.value+vcoord*y.value))

    def _translate_coordinates(self, x, y):
        return x-self.params["x"].value, y-self.params["y"].value


class AnalyticalComponent(Component):
    """Class for all analytically calculated components."""
    name = "Analytical Component"
    shortname = "AnaComp"
    description = "This is the class from which all"\
                  "analytical components are derived."
    elliptic = False

    def __init__(self, **kwargs):
        """The class's constructor."""
        super().__init__(**kwargs)
        if ("elong" in kwargs) or ("pa" in kwargs) or self.elliptic:
            self.params["elong"] = Parameter(**STANDARD_PARAMETERS["elong"])
            self.params["pa"] = Parameter(**STANDARD_PARAMETERS["pa"])
            self.elliptic = True
        self._eval(**kwargs)

    def _image_function(self, xx: u.mas, yy: u.mas,
                        wavelength: Optional[u.um] = None) -> Optional[u.Quantity]:
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

    def _visibility_function(self,
                             wavelength: Optional[u.um] = None) -> np.ndarray:
        """Calculates the complex visibility of the the component's image.

        Parameters
        ----------
        wavelength : astropy.units.um, optional

        Returns
        -------
        complex_visibility_function : numpy.ndarray
        """
        return

    def calculate_image(self, dim,
                        pixel_size: Optional[float] = None,
                        wavelength: Optional[float] = None) -> u.Quantity:
        """Calculates a 2D image.

        Parameters
        ----------
        dim : float
            The dimension [px].
        pixel_size : float
            The size of a pixel [mas].
        wavelength : u.m, optional
            The wavelength.

        Returns
        -------
        image : astropy.units.Quantity
        """
        if OPTIONS["fourier.binning"] is not None:
            dim = get_binned_dimension(dim,
                                       OPTIONS["fourier.binning"])
        v = np.linspace(-0.5, 0.5, dim)
        x_arr, y_arr = self._translate_coordinates(*np.meshgrid(v, v))

        if self.elliptic:
            pa_rad = self.params["pa"]().to(u.rad).value
            xp = x_arr*np.cos(pa_rad)-y_arr*np.sin(pa_rad)
            yp = x_arr*np.sin(pa_rad)+y_arr*np.cos(pa_rad)
            x_arr, y_arr = xp*self.params["elong"].value, yp
        return self._image_function(x_arr, y_arr, wavelength)

    def calculate_complex_visibility(self,
                                     wavelength: Optional[u.m] = None) -> np.ndarray:
        """Calculates the complex visibility of the the component's image.

        Parameters
        ----------
        wavelength : astropy.units.m, optional

        Returns
        -------
        complex_visibility_function : numpy.ndarray
        """
        return self._visibility_function(wavelength)


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

    def __init__(self, **kwargs):
        """The class's constructor."""
        super().__init__(**kwargs)
        self.params["pixel_size"] = Parameter(**STANDARD_PARAMETERS["pixel_size"])
        self.params["pa"] = Parameter(**STANDARD_PARAMETERS["pa"])

        if self.elliptic:
            self.params["elong"] = Parameter(**STANDARD_PARAMETERS["elong"])
        self._eval(**kwargs)

    def _calculate_internal_grid(self) -> Tuple[u.mas, u.mas]:
        """Calculates the model grid.

        Returns
        -------
        xx : astropy.units.mas
            The x-coordinate grid.
        yy : astropy.units.mas
            The y-coordinate grid.
        """
        v = np.linspace(-0.5, 0.5, self.params["dim"].value, endpoint=False)\
            * self.params["pixel_size"]().to(u.mas)*self.params["dim"].value
        return np.meshgrid(v, v)

    def _image_function(self, xx: u.mas,
                        yy: u.mas, wavelength: u.m) -> Optional[u.Quantity]:
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
        image : astropy.units.Quantity, optional
        """
        return

    def calculate_internal_image(self,
                                 wavelength: Optional[u.m] = None
                                 ) -> u.Quantity:
        """Calculates the internal image of the component.

        Parameters
        ----------
        wavelength : astropy.units.m, optional

        Returns
        -------
        image : astropy.units.Quantity
        """
        x_arr, y_arr = self._calculate_internal_grid()
        return self._image_function(x_arr, y_arr, wavelength)


    def calculate_image(self, dim: float, pixel_size: float,
                        wavelength: Optional[u.um] = None) -> u.Quantity:
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
        v = np.linspace(-0.5, 0.5, dim)*pixel_size*dim
        x_arr, y_arr = self._translate_coordinates(*np.meshgrid(v, v))

        if self.elliptic:
            pa_rad = self.params["pa"]().to(u.rad).value
            xp = x_arr*np.cos(pa_rad)-y_arr*np.sin(pa_rad)
            yp = x_arr*np.sin(pa_rad)+y_arr*np.cos(pa_rad)
            x_arr, y_arr = xp*self.params["elong"].value, yp
        return self._image_function(x_arr, y_arr, wavelength)

    def calculate_complex_visibility(self,
                                     wavelength: Optional[u.um] = None
                                     ) -> np.ndarray:
        """Calculates the complex visibility of the the component's image.

        Parameters
        ----------
        wavelength : astropy.units.um, optional

        Returns
        -------
        complex_visibility_function : numpy.ndarray
        """
        image = self.calculate_internal_image(wavelength)
        return compute_2Dfourier_transform(image)
