from astropy.units import Quantity
from typing import List, Optional

import numpy as np
import astropy.units as u


def compute_2Dfourier_transform(image: np.ndarray,
                                pixel_size: float) -> np.ndarray:
    """Calculates the Fourier transform.

    Parameters
    ----------
    image : numpy.ndarray
    pixel_size : float

    Returns
    --------
    fourier_transform : np.ndarray
    """
    frequency_axis = np.fft.ifftshift(np.fft.fftfreq(image.shape[-1], pixel_size))
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image)))


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
