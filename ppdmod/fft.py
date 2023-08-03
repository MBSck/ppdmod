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
    return np.fft.fftshift(
        np.fft.fftfreq(dim, pixel_size.to(u.rad).value))*wavelength.to(u.m)


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


if __name__ == "__main__":
    print(get_frequency_axis(512, 0.1*u.mas, 4.5*u.um))
