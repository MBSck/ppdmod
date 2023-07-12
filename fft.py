from astropy.units import Quantity
from typing import List, Optional

import numpy as np
import astropy.units as u
from scipy import interpolate


def compute_2Dfourier_transform(image: np.ndarray,
                                ucoord: np.ndarray,
                                vcoord: np.ndarray,
                                wavelength: np.ndarray,
                                wavelength_filtered: np.ndarray,
                                pixel_size: float) -> np.ndarray:
    """Calculates the Fourier transform.

    Parameters
    ----------
    image : numpy.ndarray
    ucoord : numpy.ndarray
    vcoord : numpy.ndarray
    wavelength : numpy.ndarray
    wavelength_filtered : numpy.ndarray
    pixel_size : float

    Returns
    --------
    interpolated_fourier_transform : np.ndarray
    """
    # TODO: Make filtered and unfiltered wavelength for fitting.
    frequency_axis = np.fft.ifftshift(np.fft.fftfreq(image.shape[-1], pixel_size))
    grid = (wavelength_filtered, frequency_axis, frequency_axis)
    coordinates = np.transpose([wavelength, vcoord, ucoord])
    # TODO: Only fourier transform the last two axes here (the image itself).
    fourier_transform = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image)))
    # TODO: Check that the interpolation here works correctly
    real = interpolate.interpn(grid, np.real(fourier_transform), coordinates,
                               bounds_error=False, fill_value=None)
    imag = interpolate.interpn(grid, np.imag(fourier_transform), coordinates,
                               bounds_error=False, fill_value=None)
    return real+imag*1j

def get_amp_phase(fourier_transform: np.ndarray,
                  unwrap_phase: Optional[bool] = False,
                  period: Optional[int] = 360) -> List[Quantity]:
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
