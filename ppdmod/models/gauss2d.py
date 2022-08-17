import sys
import inspect
import numpy as np
import matplotlib.pyplot as plt

from typing import Any, Dict, List, Union, Optional

from ..functionality.fourier import FFT
from ..functionality.baseClasses import Model
from ..functionality.utilities import timeit, set_grid, set_uvcoords, mas2rad


class Gauss2D(Model):
    """Two dimensional Gauss model, FFT is also Gauss

    ...

    Attributes
    ----------
        Methods
    -------
    eval_model():
        Evaluates the model
    eval_vis2():
        Evaluates the visibilities of the model
    """
    def __init__(self, T_sub, T_eff, L_star, distance, wavelength):
        super().__init__(T_sub, T_eff, L_star, distance, wavelength)
        self.name = "2D-Gaussian"

    def eval_model(self, theta: List, mas_size: int, px_size: int,
                   sampling: Optional[int] = None) -> np.array:
        """Evaluates the model

        Parameters
        ----------
        fwhm: int | float
            The fwhm of the gaussian
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
            fwhm = theta[0]
        except:
            raise IOError(f"{self.name}.{inspect.stack()[0][3]}():"
                          " Check input arguments, theta must be of"
                          " the form [fwhm]")
        if sampling is None:
            sampling = px_size

        self._size, self._sampling, self._mas_size = px_size, sampling, mas_size
        self._radius, self._axis_mod, self._phi  = set_grid(mas_size, px_size,
                                                            sampling)

        return (1/(np.sqrt(np.pi/(4*np.log(2)))*fwhm))*np.exp((-4*np.log(2)*self._radius**2)/fwhm**2)

    def eval_vis(self, theta: np.ndarray, sampling: int,
                 wavelength: float, size: Optional[int] = 200,
                 uvcoords: Optional[np.ndarray] = None) -> np.array:
        """Evaluates the visibilities of the model

        Parameters
        ----------
        fwhm: int | float
            The diameter of the sphere
        wavelength: int
            The sampling wavelength
        sampling: int, optional
            The sampling of the uv-plane
        size: int
            The size of the (u,v)-axis
        uvcoords: List[float], optional
            If uv-coords are given, then the visibilities are calculated for
            precisely these.

        Returns
        -------
        visibility: np.array

        See also
        --------
        set_uvcoords()
        """
        try:
            fwhm = mas2rad(theta[0])
        except:
            raise IOError(f"{self.name}.{inspect.stack()[0][3]}():"
                          " Check input arguments, theta must be"
                          " of the form [fwhm]")

        self._sampling = sampling
        B, self._axis_vis  = set_uvcoords(wavelength, sampling, size,
                                          uvcoords=uvcoords)

        return np.exp(-(np.pi*fwhm*B)**2/(4*np.log(2)))

if __name__ == "__main__":
    ...

