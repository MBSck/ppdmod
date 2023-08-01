from typing import Optional, List

import astropy.units as u
import numpy as np
import pyfftw
from scipy.interpolate import interpn

from .options import OPTIONS


def compute_2Dfourier_transform(image: np.ndarray) -> np.ndarray:
    """Calculates the Fourier transform.

    Parameters
    ----------
    image : numpy.ndarray

    Returns
    --------
    interpolated_fourier_transform : np.ndarray
    """
    if OPTIONS["fourier.backend"] == "numpy":
        return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image)))

    ft_input = pyfftw.empty_aligned(image.shape, dtype="complex128")
    ft_input[:] = np.fft.ifftshift(image)
    ft_output = pyfftw.empty_aligned(image.shape, dtype="complex128")
    return np.fft.fftshift(pyfftw.FFTW(ft_input, ft_output, axes=(-2, -1))())


def get_frequency_axis(dim: int, pixel_size: float, wavelength: float) -> np.ndarray:
    """Calculates the frequency axis in meters at the observed wavelength.

    Parameters
    ----------
    dim : float
        The dimension [px].
    pixel_size : float
        The pixel size [u.rad].
    wavelength : numpy.ndarray
        The wavelength [u.m].

    Returns
    -------
    frequency_axis : numpy.ndarray
        The frequency axis [u.m].
    """
    return np.fft.fftshift(np.fft.fftfreq(dim, pixel_size))*wavelength


def interpolate_for_coordinates(fourier_transform: np.ndarray,
                                dim: float,
                                pixel_size: float,
                                ucoord: np.ndarray,
                                vcoord: np.ndarray,
                                wavelength: np.ndarray):
    """Interpolate the coordinates.

    Parameters
    ----------
    fourier_transform : numpy.ndarray
    dim : float
    pixel_size : float
    ucoord : numpy.ndarray
    vcoord : numpy.ndarray
    wavelength_axis : numpy.ndarray
    wavelength : numpy.ndarray

    Returns
    -------
    numpy.ndarray
    """
    frequency_axis = get_frequency_axis(dim, pixel_size, wavelength)
    grid = (frequency_axis, frequency_axis)
    coordinates = np.transpose([vcoord, ucoord])
    real = interpn(grid, np.real(fourier_transform), coordinates)
    imag = interpn(grid, np.imag(fourier_transform), coordinates)
    return real+imag*1j


# TODO: Look up how to calculate the closure phases from the fourier transform
# TODO: Look up how to calculate the total flux (add up the individual fluxes probs of the image)
# or just pick the zeroth frequency?


def get_amp_phase(fourier_transform: np.ndarray,
                  unwrap_phase: Optional[bool] = False,
                  period: Optional[int] = 360) -> List[u.Quantity]:
    """Gets the amplitude and the phase of the FFT

    Parameters
    ----------
    ft : np.ndarray
        The Fourier transform.

    Returns
    --------
    amp: astropy.units.Quantity
        The correlated fluxes or normed visibilities or visibilities
        squared pertaining to the function's settings
    phase: astropy.units.Quantity
        The phase information of the image after FFT
    """
    amp, phase = np.abs(fourier_transform), np.angle(fourier_transform, deg=True)
    if unwrap_phase:
        phase = np.unwrap(phase, period=period)
    return amp*u.Jy, phase*u.deg


if __name__ == "__main__":
    print(get_frequency_axis(512, 0.1*u.mas.to(u.rad), 4.5e-6))
