import os
import sys
import time
import inspect
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as c

from astropy.io import fits
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Union, Optional, Callable

def orbit_au2arc(orbit_radius: Union[int, float],
                 distance: Union[int, float]):
    """Converts the orbital radius from [au] to [arc]

    Parameters
    ----------
    orbit_radius: int | float
        The radius of the star or its orbit
    distance: int | float
        The distance to the star

    Returns
    -------
    orbit: float
        The orbit in arcseconds
    """
    return orbit_radius/distance

def m2au(radius: Union[int, float]):
    """Converts units of [m] to [au]"""
    return radius/AU_M

def m2arc(radius: float, distance: int):
    """Converts [m] to [arcsec]"""
    return orbit_au2arc(m2au(radius), distance)

def m2mas(radius: float, distance: int):
    """Converts [m] to [mas]"""
    return m2arc(radius, distance)*1000

def m2rad(radius: float, distance: int):
    """Converts [m] to [rad]"""
    return arc2rad(m2arc(radius, distance))

def arc2rad(length_in_arc: Union[int, float]):
    """Converts the orbital radius from [arcsec] to [rad]"""
    return length_in_arc*ARCSEC2RADIANS

def mas2rad(angle: Optional[Union[int, float, np.ndarray]] = None):
    """Returns a given angle in mas/rad or the pertaining scaling factor

    Parameters
    ----------
    angle: int | float | np.ndarray, optional
        The input angle(s)

    Returns
    -------
    float
        The angle in radians
    """
    if angle is None:
        return np.radians(1/3.6e6)
    return np.radians(angle/3.6e6)

def rad2mas(angle: Optional[float] = None):
    """Converts from radians to milliarcseconds"""
    return 1/mas2rad(angle)

def sr2mas(mas_size: float, sampling: int):
    """Converts the dimensions of an object from 'sr' to 'mas'. the result is
    in per pixel

    Parameters
    ----------
    mas_size: float
        The size of the image [mas]
    sampling: int
        The pixel sampling of the image
    """
    return (mas_size/(sampling*3600e3*180/np.pi))**2

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

def set_grid(mas_size: int, size: int, sampling: Optional[int] = None,
             incline_params: Optional[List[float]] = None) -> np.array:
    """Sets the size of the model and its centre. Returns the polar coordinates

    Parameters
    ----------
    mas_size: int
        Sets the size of the image [mas]
    size: int
        Sets the range of the model image and implicitly the x-, y-axis.
        Size change for simple models functions like zero-padding
    sampling: int, optional
        The pixel sampling
    incline_params: List[float], optional
        A list of the inclination parameters [axis_ratio, pos_angle] [mas, rad]

    Returns
    -------
    radius: np.array
        The radius
    xc: np.ndarray
        The x-axis used to calculate the radius
    """
    fov_scale = mas_size/size

    if sampling is None:
        sampling = size

    x = np.linspace(-size//2, size//2, sampling, endpoint=False)*fov_scale
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

        pos_angle = np.radians(pos_angle)

        xr, yr = x*np.cos(pos_angle)+y*np.sin(pos_angle),\
                (-x*np.sin(pos_angle)+y*np.cos(pos_angle))/axis_ratio
        radius = np.sqrt(xr**2+yr**2)
        axis, phi = [xr, yr], np.arctan2(xr, yr)
    else:
        radius = np.sqrt(x**2+y**2)
        axis, phi = [x, y], np.arctan2(x, y)

    return radius, axis, phi

def zoom_array(array: np.ndarray, bounds: List) -> np.ndarray :
    """Zooms in on an image by cutting of the zero-padding

    Parameters
    ----------
    array: np.ndarray
        The image to be zoomed in on
    bounds: int
        The boundaries for the zoom, the minimum and maximum

    Returns
    -------
    np.ndarray
        The zoomed in array
    """
    min_ind, max_ind = bounds
    return array[min_ind:max_ind, min_ind:max_ind]

def set_uvcoords(wavelength: float, sampling: int, size: Optional[int] = 200,
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


if __name__ == "__main__":
    ...

