from astropy.units import Quantity
from typing import List, Optional

import numpy as np
import astropy.units as u
from scipy import interpolate


def compute_2Dfourier_transform(image: np.ndarray,
                                pixel_size: float) -> np.ndarray:
    """Calculates the Fourier transform.

    Parameters
    ----------
    image : numpy.ndarray
    ucoord : numpy.ndarray
    vcoord : numpy.ndarray
    wavelengths : numpy.ndarray
    pixel_size : float

    Returns
    --------
    interpolated_fourier_transform : np.ndarray
    """
    # TODO: Make this cycle per radians, that is put the pixel_size in rad/px
    frequency_axis = np.fft.ifftshift(np.fft.fftfreq(image.shape[-1], pixel_size))
    # TODO: Only fourier transform the last two axes here (the image itself).
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image))), frequency_axis

# TODO: Fix this. Where should the frequenciy axis belong to
def interpolate_coordinates(fourier_transform: np.ndarray,
                            frequency_axis: np.ndarray,
                            ucoord: np.ndarray,
                            vcoord: np.ndarray,
                            wavelength_axis: np.ndarray,
                            wavelengths: np.ndarray):
    """Interpolate the coordinates.

    Parameters
    ----------
    fourier_transform : numpy.ndarray
    frequency_axis : numpy.ndarray
    ucoord : numpy.ndarray
    vcoord : numpy.ndarray
    wavelength_axis : numpy.ndarray
    wavelengths : numpy.ndarray

    Returns
    -------
    numpy.ndarray
    """
    # NOTE: Convert spatial frequencies from cycles/rad to cycles/meter.
    # TODO: The wavelengths need to be in u.m.
    frequency_axis = np.diff(frequency_axis)[0]*wavelengths
    grid = (wavelengths, frequency_axis, frequency_axis)
    coordinates = np.transpose([wavelength_axis, vcoord, ucoord])
    real = interpolate.interpn(grid, np.real(fourier_transform), coordinates,
                               bounds_error=False, fill_value=None)
    imag = interpolate.interpn(grid, np.imag(fourier_transform), coordinates,
                               bounds_error=False, fill_value=None)
    return real+imag*1j


# TODO: Look up how to calculate the closure phases from the fourier transform
# TODO: Look up how to calculate the total flux (add up the individual fluxes probs of the image)
# or just pick the zeroth frequency?


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
