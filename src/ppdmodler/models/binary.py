import sys
import inspect
import numpy as np
import matplotlib.pyplot as plt

from typing import Any, Dict, List, Union, Optional

from .functionality.fourier import FFT
from .functionality.baseClasses import Model
from .functionality.utilities import timeit, set_size, set_uvcoords, mas2rad


class Binary(Model):
    """..."""
    def __init__(self, T_sub, T_eff, L_star, distance, wavelength):
        super().__init__(T_sub, T_eff, L_star, distance, wavelength)
        self.name = "Binary"

    def eval_model(self, theta: List, mas_size: int, px_size: int,
                   sampling: Optional[int] = None) -> np.array:
        """Evaluates the model

        Parameters
        ----------
        q: float, optional
            The power law index
        size: int
            The size of the model image
        sampling: int, optional
            The sampling of the object-plane
        wavelength: float, optional
            The measurement wavelength
        centre: int, optional
            The centre of the model image

        Returns
        --------
        model: np.array

        See also
        --------
        set_size()
        """
        try:
            flux1, flux2, x1, y1, x2, y2 = theta
        except:
            raise IOError(f"{self.name}.{inspect.stack()[0][3]}():"
                          " Check input arguments, theta must be of"
                          " the form either [flux1, flux2, separation]")

        self._size = self._sampling = px_size
        self._mas_size = mas_size
        self._radius = np.zeros((px_size, px_size))

        centre = px_size//2
        separation = np.sqrt((x1-x2)**2+(y1-y2)**2)*self.pixel_scale

        self._radius[centre+y1, centre+x1] = flux1
        self._radius[centre+y2, centre+x2] = flux2

        return self._radius

    def eval_vis(self, theta: List, wavelength: float,
                 sampling: int, size: Optional[int] = 200,
                 uvcoords: np.ndarray = None) -> np.array:
        """Evaluates the visibilities of the model

        Parameters
        ----------
        theta: List
            The parameters for the model/analytical function
        wavelength: float
            The sampling wavelength
        sampling: int
            The pixel sampling
        size: int, optional
            Sets the range of the (u,v)-plane in meters, with size being the
            longest baseline
        uvcoords: List[float], optional
            If uv-coords are given, then the visibilities are calculated for
            precisely these.

        Returns
        -------
        complex_visibilities: np.array
            The Fourier transform of the intensity distribution

        See also
        --------
        set_uvcoords()
        """
        try:
            flux1, flux2, x1, y1, x2, y2 = theta
        except:
            raise IOError(f"{self.name}.{inspect.stack()[0][3]}():"
                          " Check input arguments, theta must be of"
                          " the form [flux1, flux2, separation]")

        sep_vec = np.array([x1-x2, y1-y2])
        x1, y1, x2, y2 = map(lambda x: mas2rad(x*self.pixel_scale), [x1, y1, x2, y2])

        B, self._axis_vis = set_uvcoords(wavelength, sampling, size, B=True)

        global axis1, axis2
        u, v = axis1, axis2 = self._axis_vis
        flux1_contribution = flux1*np.exp(2*np.pi*-1j*(u*x1+v*y1))
        flux2_contribution = flux2*np.exp(2*np.pi*-1j*(u*x2+v*y2))

        return flux1_contribution + flux2_contribution


if __name__ == "__main__":
    ...

