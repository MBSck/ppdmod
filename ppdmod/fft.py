import astropy.units as u
import numpy as np
import pyfftw
from scipy.interpolate import interpn

from .options import OPTIONS


# NOTE: Combining real and conjugate may yield an error with the sign in the
# the middle of the fft.
def compute_2Dfourier_transform(image: np.ndarray) -> np.ndarray:
    """Calculates the Fourier transform.

    Parameters
    ----------
    image : numpy.ndarray

    Returns
    --------
    interpolated_fourier_transform : np.ndarray
    """
    if isinstance(image, u.Quantity):
        image = image.value
    if OPTIONS["fourier.backend"] == "numpy":
        real_fft = np.fft.rfft2(np.fft.ifftshift(image))
    else:
        fft_input = pyfftw.empty_aligned(image.shape, dtype=str(image.dtype))
        # fft_output = pyfftw.empty_aligned(
        #     (image.shape[0], image.shape[1]//2+1), dtype="complex128")
        fft_input[...] = np.fft.fftshift(image.copy())
        fft_object = pyfftw.builders.rfft2(fft_input)
        # fft_object = pyfftw.FFTW(
        #     fft_input, fft_output,
        #     axes=(0, 1), direction="FFTW_FORWARD")
        real_fft = fft_object()
    conjugated_fft = np.conjugate(np.vstack(
        (real_fft[0, 1:-1][::-1], real_fft[1:, 1:-1][::-1, ::-1])))
    combined_fft = np.concatenate((real_fft, conjugated_fft), axis=1)
    return np.fft.fftshift(combined_fft)


def get_frequency_axis(dim: int,
                       pixel_size: u.mas, wavelength: u.um) -> np.ndarray:
    """Calculates the frequency axis in meters at the observed wavelength.

    Parameters
    ----------
    dim : float
        The dimension [px].
    pixel_size : astropy.units.mas
        The pixel size.
    wavelength : astropy.units.um

    Returns
    -------
    frequency_axis : astropy.units.m/astropy.units.rad
    """
    pixel_size = pixel_size.to(u.rad).value
    if OPTIONS["fourier.binning"] is not None:
        pixel_size *= 2**OPTIONS["fourier.binning"]
    return np.fft.fftshift(np.fft.fftfreq(dim, pixel_size))*wavelength.to(u.m)


# TODO: Look up how to calculate the closure phases from the fourier transform
def interpolate_coordinates(fourier_transform: np.ndarray,
                            dim: float,
                            pixel_size: u.mas,
                            ucoord: np.ndarray,
                            vcoord: np.ndarray,
                            wavelength: np.ndarray):
    """Interpolate the coordinates.

    Parameters
    ----------
    fourier_transform : numpy.ndarray
    dim : float
    pixel_size : astropy.units.mas
    ucoord : astropy.units.m
    vcoord : astropy.units.m
    wavelength : astropy.units.um

    Returns
    -------
    numpy.ndarray
    """
    intp_setting = {"method": "linear",
                    "fill_value": None, "bounds_error": True}
    frequency_axis = get_frequency_axis(dim, pixel_size, wavelength)
    grid = (frequency_axis, frequency_axis)
    coordinates = np.transpose([vcoord, ucoord])
    real = interpn(
        grid, np.real(fourier_transform), coordinates, **intp_setting)
    imag = interpn(
        grid, np.imag(fourier_transform), coordinates, **intp_setting)
    return real+1j*imag
