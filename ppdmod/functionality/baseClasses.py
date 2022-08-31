import numpy as np
import astropy.units as u

from astropy.modeling import models
from typing import Union, Optional
from abc import ABCMeta, abstractmethod

from .utils import sublimation_radius, sr2mas, stellar_radius_pc, sublimation_temperature

# TODO: Implement FFT as a part of the model class?

class Model(metaclass=ABCMeta):
    """Abstract metaclass that initiates the models

    ...

    Methods
    -------
    eval_model():
        Evaluates the model
    eval_vis2():
        Evaluates the visibilities of the model
    """
    def __init__(self, T_sub, T_eff, L_star, distance, wavelength):
        self.name = ""
        self._radius  = None
        self._size, self._sampling, self._mas_size = 0, 0, 0
        self._axis_mod, self._axis_vis = [], []
        self._phi = []
        self._inner_r = 0
        self._max_sub_flux = None

        self.T_sub, self.T_eff, self.L_star, self.d, self.wl = T_sub, T_eff, \
                L_star, distance, wavelength

        self._r_sub = sublimation_radius(self.T_sub, self.L_star, self.d)
        self._stellar_radius = stellar_radius_pc(self.T_eff, self.L_star)
        self._stellar_radians = plancks_law_nu(self.T_eff, self.wl)
        self._stellar_flux = np.pi*(self._stellar_radius/self.d)**2*\
                self._stellar_radians*1e26

    @property
    def pixel_scale(self):
        """Calculates the pixel scale for max per px"""
        return self._mas_size/self._sampling

    @property
    def stellar_flux(self):
        """Get the stellar flux scaled for the pixels"""
        return self._stellar_flux

    def get_total_flux(self, *args) -> np.ndarray:
        """Sums up the flux from [Jy/px] to [Jy]"""
        return np.sum(self.get_flux(*args))

    @property
    def r_sub(self):
        """Calculates the sublimation radius"""
        return self._r_sub

    @r_sub.setter
    def r_sub(self, value):
        """Sets the sublimation radius"""
        self._r_sub = value

    def get_flux(self, tau_0: float, q: float, p: float,
                 r_sub: Optional[Union[int, float]] = None) -> np.array:
        """Calculates the total flux of the model

        Parameters
        ----------
        tau_0: float
            The optical depth of the disk, value between 0-1, which 1 being
            a perfect black body
        q: float
            The power law exponent of temperature
        p: float
            The power law exponent of optical depth
        r_sub: int | float, optional
            The inner radius used to calculate the inner/sublimation
            temperature, if provided

        Returns
        -------
        flux: np.ndarray
        """
        with np.errstate(divide='ignore'):
            if r_sub is not None:
                sub_temperature = sublimation_temperature(r_sub, self.L_star, self.d)
            else:
                sub_temperature = self._r_sub

            temperature = models.PowerLaw1D(self._radius, self._r_sub, q, self.T_sub)
            tau = models.PowerLaw1D(self._radius, self._r_sub, p, tau_0)
            blackbody = models.BlackBody(temperature=temperature*u.K)

            flux = blackbody(self.wl)
            flux *= (1-np.exp(-tau))*sr2mas(self._mas_size, self._sampling)
            flux[np.where(np.isnan(flux))],flux[np.where(np.isinf(flux))] = 0., 0.
            return flux*1e26

    @abstractmethod
    def eval_model() -> np.array:
        """Evaluates the model image
        Convention to put non fitting parameters at end of *args.

        Returns
        --------
        np.array
            Two dimensional array that can be plotted with plt.imread()
        """
        pass

    @abstractmethod
    def eval_vis() -> np.array:
        """Evaluates the visibilities of the model.
        Convention to put non fitting parameters at end of *args.

        Returns
        -------
        np.array
            Two dimensional array that can be plotted with plt.imread()
        """
        pass

if __name__ == "__main__":
    ...

