from typing import Optional

import astropy.units as u
import numpy as np

from fft import compute_2Dfourier_transform
from options import OPTIONS
from parameter import STANDARD_PARAMETERS, Parameter
from utils import pad_image, rebin_image


class Component:
    name = "Generic component"
    shortname = "GenComp"
    description = "This is the class from which all components are derived."

    def __init__(self, **kwargs):
        """The class's constructor."""
        self.name = None
        self._wl = None
        self._allowExternalRotation = True

        self.params = {}
        self.params["x"] = Parameter(**STANDARD_PARAMETERS["x"])
        self.params["y"] = Parameter(**STANDARD_PARAMETERS["y"])
        self._eval(**kwargs)

    def _eval(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key].value = value

    def _translate_fourier_transform(self, ucoord, vcoord, wl):
        x = self.params["x"](wl)*self.params["x"].unit.to(u.rad)
        y = self.params["y"](wl)*self.params["y"].unit.to(u.rad)
        return np.exp(-2*1j*np.pi*(ucoord*x+vcoord*y))

    def _translate_coordinates(self, x, y, wl):
        return x-self.params["x"](wl), y-self.params["y"](wl)


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

    def _visibility_function(self, ucoord, vcoord, rho, wl):
        return

    def _calculate_image(self, dim, pixSize, wl=None):
        wl = np.array(wl).flatten()
        dims = (wl.size, dim, dim)

        v = np.linspace(-0.5, 0.5, dim)
        vx, vy = np.meshgrid(v, v)
        vx_arr = np.tile(vx[None, None, :, :], (wl.size, 1, 1))
        vy_arr = np.tile(vy[None, None, :, :], (wl.size, 1, 1))
        wl_arr = np.tile(wl[None, :, None, None], (1, dim, dim))
        x_arr, y_arr, wl_arr = map(lambda x: x.flatten(), [vx_arr, vy_arr, wl_arr])
        x_arr, y_arr = self._translate_coordinates(x_arr, y_arr, wl_arr)

        if self.elliptic:
            pa_rad = (self.params["pa"](wl_arr)) * \
                self.params["pa"].unit.to(u.rad)
            xp = x_arr*np.cos(pa_rad)-y_arr*np.sin(pa_rad)
            yp = x_arr*np.sin(pa_rad)+y_arr*np.cos(pa_rad)
            x_arr, y_arr = xp*self.params["elong"](wl_arr), yp

        return self._image_function(x_arr.reshape(dims),
                                    y_arr.reshape(dims), wl_arr.reshape(dims))

    def calculate_complex_visibility(self, ucoord, vcoord, wl=None):
        if self.elliptic:
            pa_rad = (self.params["pa"](wl))*self.params["pa"].unit.to(u.rad)
            co, si = np.cos(pa_rad). np.sin(pa_rad)
            fxp, fyp = ucoord*co-vcoord*si, ucoord*si+vcoord*co
            rho = np.sqrt(fxp**2/self.params["elong"](wl)**2+fyp**2)
        else:
            fxp, fyp = ucoord, vcoord
            rho = np.sqrt(fxp**2+fyp**2)

        return self._visibility_function(fxp, fyp, rho, wl)\
            * self._translate_fourier_transform(ucoord, vcoord, wl)


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
        self._pixel_size = None, None

        self.params["dim"] = Parameter(**STANDARD_PARAMETERS["dim"])
        self.params["pixSize"] = Parameter(**STANDARD_PARAMETERS["pixSize"])

        self.params["pa"] = Parameter(**STANDARD_PARAMETERS["pa"])
        if self.elliptic:
            self.params["elong"] = Parameter(**STANDARD_PARAMETERS["elong"])
        self._eval(**kwargs)

    @property
    def pixel_size(self):
        """Gets the pixel size of the object [rad]."""
        if "pixSize" in self.params and self._pixel_size is None:
            self._pixel_size = self.params["pixSize"].value\
                * self.params["pixSize"].unit.to(u.rad)
        return self._pixel_size

    @pixel_size.setter
    def pixel_size(self, value):
        """Sets the pixel size of the object."""
        self._pixel_size = value

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
                                     ucoord: np.ndarray,
                                     vcoord: np.ndarray,
                                     wl: Optional[np.ndarray] = None
                                     ) -> np.ndarray:
        if wl is None:
            wl = ucoord*0

        if self._wl is None:
            wl0 = np.sort(np.unique(wl))
        else:
            wl0 = self._wl

        dim = self.params["dim"].value
        image = self.calculate_internal_image(wl0).reshape(wl0.size, dim, dim)
        if OPTIONS["FTBinningFactor"] is not None:
            image = rebin_image(image, OPTIONS["FTBinningFactor"])
        image = pad_image(image)

        if self._allowExternalRotation:
            pa_rad = (self.params["pa"](wl)) * \
                self.params["pa"].unit.to(u.rad)
            co, si = np.cos(pa_rad), np.sin(pa_rad)
            fxp, fyp = ucoord*co-vcoord*si, ucoord*si+vcoord*co
            vcoord = fyp
            if self.elliptic:
                ucoord = fxp/self.params["elong"](wl)
            else:
                ucoord = fxp

        return compute_2Dfourier_transform(image, self.pixel_size)\
            * self._translate_fourier_transform(ucoord, vcoord, wl)

    def calculate_image(self, dim: float, pixSize: float,
                        wl: Optional[np.ndarray] = None) -> np.ndarray:
        if wl is None:
            wl = 0

        wl = np.array(wl).flatten()
        dims = (wl.size, dim, dim)

        v = np.linspace(-0.5, 0.5, dim)*pixSize*dim
        vx, vy = np.meshgrid(v, v)
        vx_arr = np.tile(vx[None, :, :], (wl.size, 1, 1))
        vy_arr = np.tile(vy[None, :, :], (wl.size, 1, 1))
        wl_arr = np.tile(wl[:, None, None], (1, dim, dim))

        x_arr, y_arr, wl_arr = map(lambda x: x.flatten(),
                                   [vx_arr, vy_arr, wl_arr])
        x_arr, y_arr = self._translate_coordinates(x_arr, y_arr, wl_arr)

        if self.elliptic:
            pa_rad = (self.params["pa"](wl_arr)) * \
                self.params["pa"].unit.to(u.rad)
            xp = x_arr*np.cos(pa_rad)-y_arr*np.sin(pa_rad)
            yp = x_arr*np.sin(pa_rad)+y_arr*np.cos(pa_rad)
            x_arr, y_arr = xp*self.params["elong"](wl_arr), yp

        im = self._image_function(x_arr, y_arr, wl_arr)
        im = im.reshape(dims)
        return im
