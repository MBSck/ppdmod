import sys
import numpy as np
import matplotlib.pyplot as plt
import inspect

from typing import Any, Dict, List, Union, Optional

from src.functionality.fourier import FFT
from src.functionality.baseClasses import Model


# TODO: Implement flux for centre of picture

class Delta(Model):
    """Delta function/Point source model

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
        self.name = "Delta"

    def eval_model(self, mas_size: int, px_size: int) -> np.array:
        """Evaluates the model

        Parameters
        ----------
        mas_size: int
            The size of the FOV
        px_size: int
            The size of the model image

        Returns
        --------
        model: np.array
        """
        self._size = self._sampling = px_size
        self._mas_size = mas_size
        self._radius = np.zeros((px_size, px_size))
        self._radius[px_size//2, px_size//2] = 1
        return self._radius

    def eval_vis(self, theta: List, sampling: int) -> np.array:
        """Evaluates the visibilities of the model

        Parameters
        ----------
        flux: float
            The flux of the object
        sampling: int
            The sampling of the uv-plane

        Returns
        -------
        visibility: np.array
        """
        try:
            flux = float(theta[0])
        except:
            raise IOError(f"{self.name}.{inspect.stack()[0][3]}():"
                          " Check input arguments, theta must"
                          " be of the form [flux]")

        self._sampling = self._size = sampling

        return flux*np.ones((sampling, sampling))

if __name__ == "__main__":
    wavelength = 8e-6
    d = Delta(1500, 7900, 19, 140, 8e-6)
    d_model = d.eval_model(10, 129)
    fft = FFT(d_model, wavelength, d.pixel_scale, 3)
    fft.plot_amp_phase()

