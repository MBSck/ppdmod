import os
import sys
import time
import inspect
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as c

from astropy.io import fits
from astropy.modeling import models
from pathlib import Path
from typing import Any, Dict, List, Union, Optional, Callable

# TODO: Make function that contains all data
# TODO: Implement FFT as a part of the base_model_class
# TODO: Check all imports, if relevant
# TODO: Check all imports, if relevant

class Model:
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

    def azimuthal_modulation(image, polar_angle: Union[float, np.ndarray],
                             modulation_angle: float,
                             amplitude: int  = 1) -> Union[float, np.ndarray]:
        """Azimuthal modulation of an object

        Parameters
        ----------
        polar_angle: float | np.ndarray
            The polar angle of the x, y-coordinates
        amplitude: int
            The 'c'-amplitude

        Returns
        -------
        azimuthal_modulation: float | np.ndarray
        """
        # TODO: Implement Modulation field like Jozsef?
        modulation_angle = np.radians(modulation_angle)
        total_mod = (amplitude*np.cos(polar_angle-modulation_angle))
        image *= np.array(1 + total_mod)
        image[image < 0.0] = 0.0
        return image

    def set_grid(mas_size: int, size: int, pixel_sampling: Optional[int] = None,
                 incline_params: Optional[List[float]] = None) -> np.array:
        """Sets the size of the model and its centre. Returns the polar coordinates

        Parameters
        ----------
        mas_size: int
            Sets the size of the image [astropy.units.mas]
        size: int
            Sets the range of the model image and implicitly the x-, y-axis.
            Size change for simple models functions like zero-padding
        pixel_sampling: int, optional
            The pixel sampling [px]
        incline_params: List[float], optional
            A list of the inclination parameters [axis_ratio, pos_angle]
            [None, astropy.units.rad]

        Returns
        -------
        radius: np.array
            The radius [astropy.units.mas/px]
        xc: np.ndarray
            The x-axis used to calculate the radius [astropy.units.mas]
        """
        # TODO: Add this to the abstract model baseclass
        pixel_sampling = size if pixel_sampling is None else pixel_sampling
        fov_scale = (mas_size*u.mas)/pixel_sampling

        x = np.linspace(-size//2, size//2, pixel_sampling, endpoint=False)*fov_scale
        y = x[:, np.newaxis]

        if incline_params:
            try:
                axis_ratio, pos_angle = incline_params
            except:
                raise IOError(f"{inspect.stack()[0][3]}(): Check input"
                              " arguments, ellipsis_angles must be of the"
                              " form [axis_ratio, pos_angle]")

            if axis_ratio < 1.:
                raise ValueError("The axis_ratio has to be bigger than 1.")

            if (pos_angle < 0) or (pos_angle > 180):
                raise ValueError("The positional angle must be between [0, 180]")

            pos_angle = (np.radians(pos_angle)*u.rad).to(u.mas)

            xr, yr = x*np.cos(pos_angle)+y*np.sin(pos_angle),\
                    (-x*np.sin(pos_angle)+y*np.cos(pos_angle))/axis_ratio
            radius = np.sqrt(xr**2+yr**2)
            axis, phi = [xr, yr], np.arctan2(xr, yr)*u.rad
        else:
            radius = np.sqrt(x**2+y**2)
            axis, phi = [x, y], np.arctan2(x, y)

        return radius, axis, phi

    def set_uv_grid(wavelength: float, sampling: int, size: Optional[int] = 200,
                    angles: List[float] = None, uvcoords: np.ndarray = None,
                    B: Optional[bool] = True) -> np.array:
        """Sets the uv coords for visibility modelling

        Parameters
        ----------
        wavelength: float
            The wavelength the (u,v)-plane is sampled at
        sampling: int
            The pixel sampling
        size: int, optional
            Sets the range of the (u,v)-plane in meters, with size being the
            longest baseline
        angles: List[float], optional
            A list of the three angles [ellipsis_angle, pos_angle inc_angle]
        uvcoords: List[float], optional
            If uv-coords are given, then the visibilities are calculated for
        B: bool, optional
            Returns the baseline vector if toggled true, else the r vector

        Returns
        -------
        baselines: ArrayLike
            The baselines for the uvcoords
        uvcoords: ArrayLike
            The axis used to calculate the baselines
        """
        # TODO: Add this to the abstract model baseclass
        if uvcoords is None:
            axis = np.linspace(-size, size, sampling, endpoint=False)

            # Star overhead sin(theta_0)=1 position
            u, v = axis/wavelength, axis[:, np.newaxis]/wavelength

        else:
            axis = uvcoords/wavelength
            u, v = np.array([i[0] for i in uvcoords]), \
                    np.array([i[1] for i in uvcoords])

        if angles is not None:
            try:
                if len(angles) == 1:
                    pos_angle = np.radians(angles[0])
                    ur, vr = u*np.cos(pos_angle)+v*np.sin(pos_angle), \
                            v*np.cos(pos_angle)-u*np.sin(pos_angle)
                    r = np.sqrt(ur**2+vr**2)
                    B_vec = r*wavelength if B else r
                else:
                    axis_ratio = angles[0]
                    pos_angle, inc_angle = map(lambda x: np.radians(x), angles[1:])

                    ur, vr = u*np.cos(pos_angle)+v*np.sin(pos_angle), \
                            (v*np.cos(pos_angle)-u*np.sin(pos_angle))/axis_ratio
                    r = np.sqrt(ur**2+vr**2*np.cos(inc_angle)**2)
                    B_vec = r*wavelength if B else r

                axis = [ur, vr]
            except:
                raise IOError(f"{inspect.stack()[0][3]}(): Check input"
                              " arguments, ellipsis_angles must be of the form"
                              " either [pos_angle] or "
                              " [ellipsis_angle, pos_angle, inc_angle]")

        else:
            r = np.sqrt(u**2+v**2)
            B_vec = r*wavelength if B else r
            axis = [u, v]

        return B_vec, axis

    def stellar_radius_pc(T_eff: int, L_star: int):
        """Calculates the stellar radius from its attributes and converts it from
        m to parsec

        Parameters
        ----------
        T_eff: int
            The star's effective temperature [K]
        L_star: int
            The star's luminosity [L_sun]

        Returns
        -------
        stellar_radius: float
            The star's radius [pc]
        """
        stellar_radius_m = np.sqrt((L_star*c.L_sun)/(4*np.pi*c.sigma_sb*T_eff**4))
        return stellar_radius_m/PARSEC2M

    def sublimation_temperature(r_sub: float, L_star: int, distance: int):
        """Calculates the sublimation temperature at the inner rim of the disk

        Parameters
        ----------
        r_sub: float
            The sublimation radius [mas]
        L_star: int
            The star's luminosity in units of nominal solar luminosity
        distance: int
            Distance in parsec

        Returns
        -------
        T_sub: float
            The sublimation temperature [K]
        """
        r_sub /= m2mas(1, distance)
        return ((L_star*c.L_sun)/(4*np.pi*c.k_B*r_sub**2))**(1/4)

    def sublimation_radius(T_sub: int, L_star: int, distance: int):
        """Calculates the sublimation radius of the disk

        Parameters
        ----------
        T_sub: int
            The sublimation temperature of the disk. Usually fixed to 1500 K
        L_star: int
            The star's luminosity in units of nominal solar luminosity
        distance: int
            Distance in parsec

        Returns
        -------
        R_sub: int
            The sublimation_radius [mas]
        """
        sub_radius_m = np.sqrt((L_star*c.L_sun)/(4*np.pi*c.sigma_sb*T_sub**4))
        return m2mas(sub_radius_m, distance)

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

    def eval_model() -> np.array:
        """Evaluates the model image
        Convention to put non fitting parameters at end of *args.

        Returns
        --------
        np.array
            Two dimensional array that can be plotted with plt.imread()
        """
        pass

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

