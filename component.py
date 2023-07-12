from typing import Optional

import astropy.units as u
import numpy as np

from fft import compute_2Dfourier_transform
from options import OPTIONS
from parameter import STANDARD_PARAMETERS, Parameter
from utils import rebin_image, pad_image, get_next_power_of_two


class NumericalComponent:
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
        """Create and initiliaze a new instance of the oimComponent class.

        All components have at least three parameters the position
        x and y and their flux f
        """
        self.name = None
        self._wl = None, None
        self._dim, self._pixel_size = None, None
        self._allowExternalRotation = True

        self.params = {}
        self.params["x"] = Parameter(**STANDARD_PARAMETERS["x"])
        self.params["y"] = Parameter(**STANDARD_PARAMETERS["y"])
        self.params["f"] = Parameter(**STANDARD_PARAMETERS["f"])
        self.params["dim"] = Parameter(**STANDARD_PARAMETERS["dim"])
        self.params["fov"] = Parameter(**STANDARD_PARAMETERS["fov"])
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

    @property
    def dim(self):
        """Gets the dimension of the object."""
        if self._dim is None:
            if self.params["fov"] != 0:
                self._dim = get_next_power_of_two(self.params["fov"].value /
                                                  self.params["pixSize"].value)
                if self.params["dim"].value != self._dim:
                    self.params["dim"].value = self._dim
            else:
                self._dim = self.params["dim"].value
        return self._dim

    def _eval(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.params.keys():
                self.params[key].value = value

    def _translate_fourier_transform(self, ucoord, vcoord, wl):
        x = self.params["x"](wl)*self.params["x"].unit.to(u.rad)
        y = self.params["y"](wl)*self.params["y"].unit.to(u.rad)
        return np.exp(-2*1j*np.pi*(ucoord*x+vcoord*y))

    def _translate_coordinates(self, x, y, wl):
        return x-self.params["x"](wl), y-self.params["y"](wl)

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
        v = np.linspace(-0.5, 0.5, self.dim)*pix*self.dim
        return v, v[:, None]

    def _calculate_image(self, x: np.ndarray, y: np.ndarray, wl: np.ndarray):
        return

    def calculate_internal_image(self, wl: np.ndarray):
        x_arr, y_arr = self._calculate_internal_grid()
        return self._calculate_image(x_arr, y_arr, wl)

    def calculate_visibility_function(self,
                                      ucoord: np.ndarray,
                                      vcoord: np.ndarray,
                                      wl: Optional[np.ndarray] = None):
        if wl is None:
            wl = ucoord*0

        if self._wl is None:
            wl0 = np.sort(np.unique(wl))
        else:
            wl0 = self._wl

        image = self.getInternalImage(wl0).reshape(wl0.size, self.dim, self.dim)
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

        # TODO: Implement custom fft here
        fourier_transform = compute_2Dfourier_transform(image, ucoord, vcoord,
                                                        wl, wl, self.pixel_size)

        translate_factor = self._translate_fourier_transform(
            ucoord, vcoord, wl)*self.params["f"](wl)

        return fourier_transform*translate_factor*self.params["f"](wl)

    def calculate_image(self, dim: float, pixSize: float,
                        wl: Optional[np.ndarray] = None):
        if wl is None:
            wl = 0

        wl = np.array(wl).flatten()
        nwl = wl.size
        dims = (nwl, dim, dim)

        v = np.linspace(-0.5, 0.5, dim)
        vx, vy = np.meshgrid(v, v)
        vx_arr = np.tile(vx[None, :, :], (nwl, 1, 1))
        vy_arr = np.tile(vy[None, :, :], (nwl, 1, 1))
        wl_arr = np.tile(wl[:, None, None], (1, dim, dim))

        x_arr = (vx_arr*pixSize*dim).flatten()
        y_arr = (vy_arr*pixSize*dim).flatten()
        wl_arr = wl_arr.flatten()

        x_arr, y_arr = self._translate_coordinates(x_arr, y_arr, wl_arr)

        if self.elliptic:
            pa_rad = (self.params["pa"](wl_arr)) * \
                self.params["pa"].unit.to(u.rad)

            xp = x_arr*np.cos(pa_rad)-y_arr*np.sin(pa_rad)
            yp = x_arr*np.sin(pa_rad)+y_arr*np.cos(pa_rad)
            x_arr, y_arr = xp*self.params["elong"](wl_arr), yp

        im = self._calculate_image(x_arr, y_arr, wl_arr)
        im = im.reshape(dims)
        return im


if __name__ == "__main__":
    test = NumericalComponent(x=10, y=5, fov=100, pixSize=0.1)
    breakpoint()
