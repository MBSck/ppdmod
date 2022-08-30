import sys
import inspect
import numpy as np
import matplotlib.pyplot as plt

from scipy.special import j1
from typing import Any, Dict, List, Union, Optional

from ..functionality.fourier import FFT
from ..functionality.baseClasses import Model
from ..functionality.utils import set_grid, set_uvcoords, mas2rad


class UniformDisk(Model):
    """Uniformly bright disc model

    ...

    Methods
    -------
    eval_model():
        Evaluates the model
    eval_vis2():
        Evaluates the visibilities of the model
    """
    def __init__(self, T_sub, T_eff, L_star, distance, wavelength):
        super().__init__(T_sub, T_eff, L_star, distance, wavelength)
        self.name = "Uniform Disk"

    def eval_model(self, theta: List, mas_size: int,
                   px_size: int, sampling: Optional[int] = None) -> np.ndarray:
        """Evaluates the model

        Parameters
        ----------
        theta: List
            The list of the input parameters (diameter, axis_ratio, pos_angle)
        mas_size: int
            The field of view in mas
        px_size: int
            The size of the model image
        sampling: int, optional
            The sampling of the object-plane

        Returns
        --------
        image: np.ndarray
            The image of the model

        See also
        --------
        set_size()
        """
        try:
            diameter, axis_ratio, pos_angle = theta
        except:
            raise IOError(f"{self.name}.{inspect.stack()[0][3]}():"
                          " Check input arguments, theta must be of"
                          " the form [diameter, axis_ratio, pos_angle]")

        if not sampling:
            sampling = px_size

        self._size, self._sampling, self._mas_size = px_size, sampling, mas_size
        self._radius, self._axis_mod, self._phi = set_grid(mas_size, px_size, sampling,
                                                     [axis_ratio, pos_angle])

        image = np.zeros((sampling, sampling))
        image[self._radius < diameter//2] = 1.
        self._radius[image == 0] = 0

        return image

    def eval_vis(self, theta: List, sampling: int, wavelength:
                 float, size: Optional[int] = 200,
                 uvcoords: np.ndarray = None) -> np.ndarray:
        """Evaluates the visibilities of the model

        Parameters
        ---------
        diameter: int
            The diameter of the sphere
        sampling: int
            The sampling of the uv-plane
        wavelength: float
            The sampling wavelength
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
            diameter = mas2rad(theta[0])
        except:
            raise IOError(f"{self.name}.{inspect.stack()[0][3]}():"
                          " Check input arguments, theta must be of"
                          " the form [diameter]")

        self._sampling, self._wavelength = sampling, wavelength
        B, self._axis_vis = set_uvcoords(wavelength, sampling, size,
                                         uvcoords=uvcoords, B=False)
        return 2*j1(np.pi*diameter*B)/(np.pi*diameter*B)

if __name__ == "__main__":
    ...

