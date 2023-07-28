from typing import Optional

import astropy.units as u
import numpy as np

from .fft import compute_2Dfourier_transform
from .options import OPTIONS
from .parameter import STANDARD_PARAMETERS, Parameter
from .utils import pad_image, get_binned_dimension


class Component:
    name = "Generic component"
    shortname = "GenComp"
    description = "This is the class from which all components are derived."

    def __init__(self, **kwargs):
        """The class's constructor."""
        self.name = None
        self._allowExternalRotation = True

        self.params = {}
        self.params["x"] = Parameter(**STANDARD_PARAMETERS["x"])
        self.params["y"] = Parameter(**STANDARD_PARAMETERS["y"])
        self.params["dim"] = Parameter(**STANDARD_PARAMETERS["dim"])
        self._eval(**kwargs)

    def _eval(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.params:
                if isinstance(value, Parameter):
                    self.params[key] = value
                else:
                    self.params[key].value = value

    def _translate_fourier_transform(self, ucoord, vcoord):
        x = self.params["x"].value*self.params["x"].unit.to(u.rad)
        y = self.params["y"].value*self.params["y"].unit.to(u.rad)
        return np.exp(-2*1j*np.pi*(ucoord*x+vcoord*y))

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

    def _image_function(self, xx, yy, wl):
        return

    def _visibility_function(self, wl):
        return

    def calculate_image(self, dim, pixSize, wl=None):
        if OPTIONS["fourier.binning"] is not None:
            dim = get_binned_dimension(dim,
                                       OPTIONS["fourier.binning"])
        v = np.linspace(-0.5, 0.5, dim)
        x_arr, y_arr = self._translate_coordinates(*np.meshgrid(v, v))

        if self.elliptic:
            pa_rad = (self.params["pa"].value) * \
                self.params["pa"].unit.to(u.rad)
            xp = x_arr*np.cos(pa_rad)-y_arr*np.sin(pa_rad)
            yp = x_arr*np.sin(pa_rad)+y_arr*np.cos(pa_rad)
            x_arr, y_arr = xp*self.params["elong"].value, yp
        return self._image_function(x_arr, y_arr, wl)

    def calculate_complex_visibility(self, wl=None):
        return self._visibility_function(wl)


class NumericalComponent(Component):
    """Base class with increased computational performance for numerical
    calculations.

    Parameters
    ----------

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
        self.params["pixSize"] = Parameter(**STANDARD_PARAMETERS["pixSize"])
        self.params["pa"] = Parameter(**STANDARD_PARAMETERS["pa"])

        if self.elliptic:
            self.params["elong"] = Parameter(**STANDARD_PARAMETERS["elong"])
        self._eval(**kwargs)

    def _calculate_internal_grid(self) -> np.ndarray:
        """Calculates the model grid.

        In case of 1D it is a radial profile and in 2D it is a grid.

        Parameters
        ----------

        Returns
        -------
        """
        pix = self.params["pixSize"].value * \
            self.params["pixSize"].unit.to(u.mas)
        v = np.linspace(-0.5, 0.5, self.params["dim"].value)\
            * pix*self.params["dim"].value
        return v, v[:, None]

    def _image_function(self, x: np.ndarray,
                        y: np.ndarray, wl: np.ndarray) -> None:
        return

    def calculate_internal_image(self, wl: np.ndarray):
        x_arr, y_arr = self._calculate_internal_grid()
        return self._image_function(x_arr, y_arr, wl)

    def calculate_complex_visibility(self,
                                     wl: Optional[np.ndarray] = None
                                     ) -> np.ndarray:
        image = self.calculate_internal_image(wl)
        if OPTIONS["fourier.padding"] is not None:
            image = pad_image(image, OPTIONS["fourier.padding"])
        return compute_2Dfourier_transform(image)

    def calculate_image(self, dim: float, pixSize: float,
                        wl: Optional[np.ndarray] = None) -> np.ndarray:
        v = np.linspace(-0.5, 0.5, dim)*pixSize*dim
        x_arr, y_arr = self._translate_coordinates(*np.meshgrid(v, v))

        if self.elliptic:
            pa_rad = (self.params["pa"].value) * \
                self.params["pa"].unit.to(u.rad)
            xp = x_arr*np.cos(pa_rad)-y_arr*np.sin(pa_rad)
            yp = x_arr*np.sin(pa_rad)+y_arr*np.cos(pa_rad)
            x_arr, y_arr = xp*self.params["elong"].value, yp
        return self._image_function(x_arr, y_arr, wl)
