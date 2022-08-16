import sys
import numpy as np
import matplotlib.pyplot as plt
import inspect

from scipy.special import j0
from typing import Any, Dict, List, Union, Optional

from src.functionality.baseClasses import Model
from src.functionality.constant import I
from src.functionality.fourier import FFT
from src.functionality.utilities import timeit, set_grid, set_uvcoords,\
        mas2rad, trunc, azimuthal_modulation

# TODO: Make the addition of the visibilities work properly, think of OOP
# abilities

class Ring(Model):
    """Infinitesimal thin ring model. Can be both cirular or an ellipsoid, i.e.
    inclined

    ...

    Methods
    -------
    eval_model():
        Evaluates the model
    eval_vis2():
        Evaluates the visibilities of the model

    See also
    --------
    set_grid()
    set_uvcoords()
    """
    def __init__(self, T_sub, T_eff, L_star, distance, wavelength):
        super().__init__(T_sub, T_eff, L_star, distance, wavelength)
        self.name = "Ring"

    def eval_model(self, theta: List, mas_size: int, px_size: int,
                   sampling: Optional[int] = None,
                   inner_radius: Optional[int] = None,
                   outer_radius: Optional[int] = None) -> np.array:
        """Evaluates the model

        Parameters
        ----------
        axis_ratio: float, optional
        pos_angle: int | float, optional
        px_size: int
            The size of the model image
        sampling: int, optional
            The sampling of the object-plane
        inner_radius: int, optional
            A set inner radius overwriting the sublimation radius

        Returns
        --------
        model: np.array

        See also
        --------
        set_grid()
        """
        try:
            axis_ratio, pos_angle = theta
        except:
            raise IOError(f"{self.name}.{inspect.stack()[0][3]}():"
                          " Check input arguments, theta must be of"
                          " the form [axis_ratio, pos_angle]")

        if sampling is None:
            self._sampling = sampling = px_size
        else:
            self._sampling = sampling

        image = np.zeros((sampling, sampling))

        if inner_radius:
            self._inner_r = self.r_sub = inner_radius
        if outer_radius:
            outer_radius = outer_radius

        self._size, self._mas_size = px_size, mas_size
        radius, self._axis_mod, self._phi = set_grid(mas_size, px_size, sampling,
                                                     [axis_ratio, pos_angle])

        if inner_radius:
            image[radius > inner_radius] = 1.
        else:
            image[radius > self.r_sub] = 1.

        if outer_radius:
            image[radius > outer_radius] = 0.

        radius[image == 0.] = 0.

        if self._radius is None:
            self._radius = radius.copy()
        else:
            self._radius += radius.copy()

        return image

    def eval_vis(self, theta: List, sampling: int,
                 wavelength: float, size: Optional[int] = 200,
                 incline_params: Optional[List] = [],
                 uvcoords: np.ndarray = None) -> np.array:
        """Evaluates the visibilities of the model

        Parameters
        ----------
        r_max: int | float
            The radius of the ring,  input in mas
        sampling: int
            The pixel sampling
        wavelength: float
            The wavelength
        size: int, optional
            The size of the (u,v)-plane
        incline_params: List, optional
            A list containing the [pos_angle, axis_ratio, inc_angle]
        uvcoords: np.ndarray, optional

        Returns
        -------
        fft: np.array
            The analytical FFT of a ring

        See also
        --------
        set_uvcoords()
        """
        try:
            r_max = mas2rad(theta[0])
        except Exception as e:
            raise IOError(f"{self.name}.{inspect.stack()[0][3]}():"
                          " Check input arguments, theta must be of the"
                          " form [r_max]")

        self._sampling, self._wavelength = sampling, wavelength
        B, self._axis_vis = set_uvcoords(wavelength, sampling, size,
                                         uvcoords=uvcoords, B=False)

        return j0(2*np.pi*r_max*B)

if __name__ == "__main__":
    wavelength, mas_fov, sampling, width, size  = 3.5e-6, 10, 2**8, 0.05, 2500
    size_Mlambda = size/(wavelength*1e6)

    r = Ring(1500, 7900, 19, 140, wavelength)
    r_model = r.eval_model([0.6, 135], mas_fov, sampling,\
                           inner_radius=1., outer_radius=2)
    plt.imshow(r_model, extent=[-size, size, -size_Mlambda, size_Mlambda],
              aspect=wavelength*1e6)
    plt.show()

