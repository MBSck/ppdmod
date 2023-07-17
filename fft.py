from astropy.units import Quantity
from typing import List, Optional

import numpy as np
import astropy.units as u
from scipy import interpolate


def compute_2Dfourier_transform(image: np.ndarray) -> np.ndarray:
    """Calculates the Fourier transform.

    Parameters
    ----------
    image : numpy.ndarray

    Returns
    --------
    interpolated_fourier_transform : np.ndarray
    """
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image)))


def get_frequency_axis(dim: int, pixel_size: float, wavelength: float) -> np.ndarray:
    """Calculates the frequency axis in meters.

    Parameters
    ----------
    dim : float
    pixel_size : float
    wavelength : numpy.ndarray

    Returns
    -------
    frequency_axis : numpy.ndarray
    """
    frequency_axis = np.fft.ifftshift(np.fft.fftfreq(dim, pixel_size*u.rad))
    # NOTE: Units of 1/m
    meter_scaling = np.diff(frequency_axis).value[0]*wavelength
    breakpoint()
    return frequency_axis


def interpolate_for_coordinates(fourier_transform: np.ndarray,
                                dim: float,
                                pixel_size: float,
                                ucoord: np.ndarray,
                                vcoord: np.ndarray,
                                wavelength_axis: np.ndarray,
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
    # NOTE: Convert spatial frequencies from cycles/rad to cycles/meter.
    frequency_axis = np.fft.ifftshift(np.fft.fftfreq(dim, pixel_size))
    frequency_axis = np.diff(frequency_axis)[0]*wavelengths
    breakpoint()
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


if __name__ == "__main__":
    get_frequency_axis(512, 0.1, 4.5e-6)
