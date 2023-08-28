import astropy.units as u
import numpy as np
import pyfftw
import scipy
from scipy.interpolate import interpn

from .options import OPTIONS


def compute_real2Dfourier_transform(image: np.ndarray) -> np.ndarray:
    """Calculates the Fourier transform.

    Parameters
    ----------
    image : numpy.ndarray

    Returns
    --------
    interpolated_fourier_transform : np.ndarray
    """
    if isinstance(image, u.Quantity):
        image = np.fft.ifftshift(image.value)
    if OPTIONS["fourier.backend"] == "numpy":
        real_fft = np.fft.rfft2(image)
    elif OPTIONS["fourier.backend"] == "scipy":
        real_fft = scipy.fft.rfft2(image)
    # TODO: Pyfftw does not seem to work as fast as it should.
    else:
        fft_input = pyfftw.empty_aligned(image.shape, dtype=str(image.dtype))
        fft_input[...] = image
        fft_object = pyfftw.builders.fft2(
            fft_input, axes=(0, 1), auto_align_input=False,
            auto_contiguous=False, avoid_copy=True)
        real_fft = fft_object()
    return np.fft.fftshift(real_fft)


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
    return np.fft.fftshift(np.fft.rfftfreq(dim, pixel_size))*wavelength.to(u.m)


# NOTE: Combining real and conjugate may yield an error with the sign in the
# the middle of the fft.
def compile_full_fourier_from_real(real_fft: np.ndarray) -> np.ndarray:
    """Compiles the full fourier transform from the real FFT.

    Notes
    -----
    Takes a lot of computational ressources. Only use for visualization.
    """
    conjugated_fft = np.conjugate(np.vstack(
        (real_fft[0, 1:-1][::-1], real_fft[1:, 1:-1][::-1, ::-1])))
    return np.concatenate((real_fft, conjugated_fft), axis=1)


# TODO: Look up how to calculate the closure phases from the fourier transform.
# Maybe faster?
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
