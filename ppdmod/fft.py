from typing import Optional, Dict

import astropy.units as u
import numpy as np
import pyfftw
import scipy
from scipy.interpolate import interpn

from .options import OPTIONS


# TODO: Pyfftw does not seem to work as fast as it should.
# Is actually slower than numpy and especially scipy.
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
        image = image.value
    image = np.fft.fftshift(image)
    if OPTIONS["fourier.backend"] == "numpy":
        rfft = np.fft.rfft2(image)
    elif OPTIONS["fourier.backend"] == "scipy":
        rfft = scipy.fft.rfft2(image)
    else:
        fft_input = pyfftw.empty_aligned(
            image.shape, dtype=str(image.dtype))
        fft_input[...] = image
        fft_object = pyfftw.builders.fft2(
            fft_input, axes=(0, 1), auto_align_input=False,
            auto_contiguous=False, avoid_copy=True)
        rfft = fft_object()
    return np.fft.ifftshift(rfft, axes=(0,))


def get_frequency_axis(
        dim: int, pixel_size: u.mas,
        wavelength: u.um, axis: Optional[int] = 1) -> np.ndarray:
    """Calculates the frequency axis in meters at the observed wavelength.

    Parameters
    ----------
    dim : float
        The dimension [px].
    pixel_size : astropy.units.mas
        The pixel size.
    wavelength : astropy.units.um
        The wavelength.
    axis : int
        The axis. Defaults to 1 (i.e., the x-axis).

    Returns
    -------
    frequency_axis : astropy.units.m/astropy.units.rad
    """
    pixel_size = pixel_size.to(u.rad).value
    if OPTIONS["fourier.binning"] is not None:
        pixel_size *= 2**OPTIONS["fourier.binning"]
    if axis == 1:
        frequency_axis = np.fft.rfftfreq(dim, pixel_size)
    else:
        frequency_axis = np.fft.fftshift(np.fft.fftfreq(dim, pixel_size))
    return frequency_axis*wavelength.to(u.m)


# NOTE: Combining real and conjugate may yield an error with the sign in the
# the middle of the fft.
def compile_full_fourier_from_real(rfft2: np.ndarray) -> np.ndarray:
    """Compiles the full fourier transform from the real FFT.

    Notes
    -----
    Takes a lot of computational ressources. Only use for visualization.
    """
    n_row, n_col = rfft2.shape
    rfft2 = np.fft.ifftshift(rfft2, axes=0)
    fft2 = np.zeros((n_row, n_row), dtype=np.complex128)
    fft2[:, :n_col] = rfft2
    fft2[0, n_col-1:] = rfft2[0, 1:][::-1].conjugate()
    fft2[1:, 1:] = np.hstack(
        (rfft2[1:, 1:], rfft2[1:, 1:][::-1, ::-1][:, 1:].conjugate()))
    return np.fft.fftshift(fft2)


def mirror_uv_coords(ucoord: np.ndarray,
                     vcoord: np.ndarray,
                     copy: Optional[bool] = True) -> Dict[str, np.ndarray]:
    """Mirrors the (u, v)-coordinates point wise if they are left
    of the origin.

    Includes a flag if they have been mirrored so they can be
    complex conjugated.

    Parameters
    ----------
    ucoord : numpy.ndarray
    vcoord : numpy.ndarray
    copy : bool, optional
        If True, the ucoord and vcoord are copied.

    Returns
    -------
    ucoord : numpy.ndarray
    vcoord : numpy.ndarray
    """
    if copy:
        ucoord, vcoord = ucoord.copy(), vcoord.copy()

    conjugates = []
    for uv_coord in np.transpose([ucoord, vcoord]):
        if uv_coord.size > 2:
            conjugate = [int(uv < 0) for uv in uv_coord[:, 0]]
        else:
            conjugate = int(uv_coord[0] < 0)
        conjugates.append(conjugate)
    conjugates = np.array(conjugates).T
    ucoord[np.where(conjugates)] = -ucoord[np.where(conjugates)]
    vcoord[np.where(conjugates)] = -vcoord[np.where(conjugates)]
    return ucoord, vcoord, conjugates


# TODO: Look up how to calculate the closure phases from the fourier transform.
# Maybe faster?
# TODO: Check if the interpolation approach works with splitting real and imag
# is it in the right order -> Ordered?
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
    if isinstance(ucoord, u.Quantity) and isinstance(vcoord, u.Quantity):
        ucoord, vcoord = ucoord.value, vcoord.value
    intp_setting = {"method": "linear",
                    "fill_value": None, "bounds_error": True}
    grid = (get_frequency_axis(dim, pixel_size, wavelength, axis=0),
            get_frequency_axis(dim, pixel_size, wavelength, axis=1))
    ucoord, vcoord, conjugates = mirror_uv_coords(ucoord, vcoord)
    coordinates = np.transpose([vcoord, ucoord])
    real = interpn(
        grid, np.real(fourier_transform), coordinates, **intp_setting)
    imag = interpn(
        grid, np.imag(fourier_transform), coordinates, **intp_setting)
    interpolated_values = real+1j*imag
    interpolated_values[np.where(conjugates.T)] =\
        np.conjugate(interpolated_values[np.where(conjugates.T)])
    return interpolated_values
