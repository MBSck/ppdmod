import astropy.units as u
import numpy as np
import pytest
from numpy.typing import ArrayLike
from scipy.special import j1

from ppdmod.fft import compute_2Dfourier_transform, interpolate_for_coordinates


DIAMETERS, DIMENSIONS =\
    [4, 10, 20], [1024, 2048, 4096]
DIAMETERS_AND_DIMS = [tuple([diameter, dim])
                      for dim in DIMENSIONS for diameter in DIAMETERS]


# TODO: Set the linespace endpoint=False for the real model as well.
def uniform_disk(diameter: float, pixel_size: float, dim: int) -> np.ndarray:
    """The surface brightness profile of a uniform disk.

    Parameters
    ----------
    diameter : float
        The uniform disk's diameter [mas].
    pixel_size : float
        The size of a pixel in the image [max/px].
    dim : int
        The image's dimension [px].

    Returns
    -------
    radial_profile : numpy.ndarray
    """
    v = np.linspace(-0.5, 0.5, dim, endpoint=False)*pixel_size*dim
    x_arr, y_arr = np.meshgrid(v, v)
    return 4*(np.hypot(x_arr, y_arr) < diameter/2)/(np.pi*diameter**2)


def uniform_disk_vis(diameter: float,
                     ucoord: ArrayLike,
                     vcoord: ArrayLike,
                     wavelength: float) -> np.ndarray:
    """Defines the complex visibility function of a uniform disk.

    Parameters
    ----------
    diameter : float
        The uniform disk's diameter [mas].
    ucoord : array_like
        The u-coordinates [m].
    vcoord : array_like
        The v-coordinates [m].
    wavelength : float
        The wavelength for the spatial frequencies' unit conversion [m].

    Returns
    -------
    complex_visibility_function : numpy.ndarray
    """
    rho = np.hypot(ucoord, vcoord)/wavelength
    diameter = diameter*u.mas.to(u.rad)
    return 2*j1(np.pi*rho*diameter)/(np.pi*diameter*rho)


@pytest.fixture
def ucoord():
    """Sets the ucoord."""
    return np.linspace(100, 160, 60)


@pytest.fixture
def vcoord(ucoord: ArrayLike):
    """Sets the vcoord."""
    return ucoord*0


@pytest.mark.parametrize("diameter, dim", DIAMETERS_AND_DIMS)
def test_interpolation(diameter: float,
                       dim: float,
                       ucoord: ArrayLike,
                       vcoord: ArrayLike):
    """This tests the interpolation of the Fourier transform,
    but more importantly, implicitly the unit conversion of the
    frequency axis."""
    pixel_size, wavelength = 0.1, 1.02322101e-05
    pixel_size_rad = pixel_size*u.mas.to(u.rad)
    ft = compute_2Dfourier_transform(uniform_disk(diameter, pixel_size, dim))
    vis = uniform_disk_vis(diameter, ucoord, vcoord, wavelength)
    interpolated_values = interpolate_for_coordinates(ft, dim, pixel_size_rad,
                                                      ucoord, vcoord,
                                                      wavelength)
    interpolated_values /= ft[dim//2, dim//2]
    interpolated_values = np.real(interpolated_values)
    assert np.allclose(vis, interpolated_values, atol=1e-2)
