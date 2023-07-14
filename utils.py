from pathlib import Path
from typing import Optional, Union, List

import astropy.units as u
import numpy as np
from astropy.modeling import models

from fluxcal import transform_spectrum_to_real_spectral_resolution
from options import OPTIONS


DL_COEFFS = {
    "low": [0.10600484,  0.01502548,  0.00294806, -0.00021434],
    "high": [-8.02282965e-05,  3.83260266e-03, 7.60090459e-05, -4.30753848e-07]
}
SPECTRAL_BINNING = {"low": 7, "high": 7}


def _make_axis(axis_end: int, steps: int):
    """Makes an axis from a negative to a postive value, with the endpoint removed to give
    an even signal for a Fourier transform

    Parameters
    ----------
    axis_end: int
        The negative/positive endpoint
    steps: int
        The steps used to get from one end to another, should be an even multiple of
        the distance between positive and negative axis_end for fourier transforms

    Returns
    -------
    axis: np.ndarray
    """
    return np.linspace(-axis_end, axis_end, steps, endpoint=False)

# TODO: Maybe even optimize calculation time further in the future
# Got it down from 3.5s to 0.66s for 3 wavelengths. It is 0.19s per wl.
def calculate_intensity(wavelength: u.um,
                        temp_profile: u.K,
                        pixel_size: Optional[float] = None) -> np.ndarray:
    """Calculates the blackbody_profile via Planck's law and the
    emissivity_factor for a given wavelength, temperature- and
    dust surface density profile.

    Parameters
    ----------
    wavelengths : astropy.units.um
        Wavelength value(s).
    temp_profile : astropy.units.K
        Temperature profile.
    pixSize: float, optional
        The pixel size [rad].

    Returns
    -------
    intensity : numpy.ndarray
        Intensity per pixel.
    """
    plancks_law = models.BlackBody(temperature=temp_profile*u.K)
    pixel_size *= u.rad
    spectral_radiance = plancks_law(wavelength*u.m).to(u.erg/(u.cm**2*u.Hz*u.s*u.rad**2))
    return (spectral_radiance*pixel_size**2).to(u.Jy).value


def pad_image(image: np.ndarray):
    """Pads an image with additional zeros for Fourier transform."""
    im0 = np.sum(image, axis=(0, 1))
    dimy = im0.shape[0]
    dimx = im0.shape[1]

    im0x = np.sum(im0, axis=1)
    im0y = np.sum(im0, axis=1)

    s0x = np.trim_zeros(im0x).size
    s0y = np.trim_zeros(im0y).size

    min_sizex = s0x*OPTIONS["FTpaddingFactor"]
    min_sizey = s0y*OPTIONS["FTpaddingFactor"]

    min_pow2x = 2**(min_sizex - 1).bit_length()
    min_pow2y = 2**(min_sizey - 1).bit_length()

    # HACK: If Image has zeros around it already then this does not work -> Rework
    if min_pow2x < dimx:
        return image

    padx = (min_pow2x-dimx)//2
    pady = (min_pow2y-dimy)//2

    return np.pad(image, ((0, 0), (0, 0), (padx, padx), (pady, pady)),
                  'constant', constant_values=0)


def rebin_image(image: np.ndarray,
                binning_factor: Optional[int] = None,
                rdim: Optional[bool] = False) -> np.ndarray:
    """Bins a 2D-image down according.

    The down binning is according to the binning factor
    in oimOptions["FTBinningFactor"]. Only accounts for
    square images.

    Parameters
    ----------
    image : numpy.ndarray
        The image to be rebinned.
    binning_factor : int, optional
        The binning factor. The default is 0
    rdim : bool
        If toggled, returns the dimension

    Returns
    -------
    rebinned_image : numpy.ndarray
        The rebinned image.
    dimension : int, optional
        The new dimension of the image.
    """
    if binning_factor is None:
        return image, image.shape[-1] if rdim else image
    new_dim = int(image.shape[-1] * 2**-binning_factor)
    binned_shape = (new_dim, int(image.shape[-1] / new_dim),
                    new_dim, int(image.shape[-1] / new_dim))
    breakpoint()
    if rdim:
        return image.reshape(binned_shape).mean(-1).mean(-2), new_dim
    return image.reshape(binned_shape).mean(-1).mean(-2)


def get_next_power_of_two(number: Union[int, float]) -> int:
    """Returns the next power of two for an integer or float input.

    Parameters
    ----------
    number : int or float
        An input number.

    Returns
    -------
    closest_power_of_two : int
        The, to the input, closest power of two.
    """
    return int(2**np.ceil(np.log2(number)))


def convert_radial_profile_to_meter(radius: Union[float, np.ndarray],
                                    distance: float,
                                    rvalue: Optional[bool] = False) -> Union[float, np.ndarray]:
    """Converts the distance from mas to meters for a radial profile around
    a star via the angular diameter small angle approximation.

    Parameters
    ----------
    radius : float or numpy.ndarray
        The radius of the object around the star [mas].
    distance : float
        The star's distance to the observer [pc].
    rvalue : bool, optional
        If toggled, returns the value witout units else returns
        an astropy.units.Quantity object. The default is False.

    Returns
    -------
    radius : float or numpy.ndarray
        The radius of the object around the star [m].

    Notes
    -----
    The formula for the angular diameter small angle approximation is

    .. math:: \\delta = \\frac{d}{D}

    where d is the distance from the star and D is the distance from the star
    to the observer and ..math::`\\delta` is the angular diameter.
    """
    radius = ((radius*u.mas).to(u.arcsec).value*distance*u.au).to(u.m)
    return radius.value if rvalue else radius


def qval_to_opacity(qval_file: Path) -> np.ndarray:
    """Reads the qval file and returns the opacity.

    Parameters
    ----------
    qval_file : pathlib.Path

    Notes
    -----
    The qval-files give the grain size in microns and the
    density in g/cm^3.
    """
    with open(qval_file, "r+", encoding="utf8") as file:
        _, grain_size, density = map(float, file.readline().strip().split())
    wavelenght_grid, qval = np.loadtxt(qval_file, skiprows=1, unpack=True)
    return wavelenght_grid, 3*qval/(4*grain_size*u.um.to(u.cm)*density)


def opacity_to_matisse_opacity(wavelength_solution: u.m,
                               opacity: Optional[np.ndarray] = None,
                               opacity_file: Optional[Path] = None,
                               qval_file: Optional[Path] = None,
                               resolution: Optional[str] = "low",
                               save_path: Optional[Path] = None):
    """Converts the opacity to the MATISSE wavelength grid."""
    wavelengths = wavelength_solution*u.m.to(u.um)
    if qval_file is not None:
        wavelength_grid, opacity = qval_to_opacity(qval_file)
    elif opacity_file is not None:
        wavelength_grid, opacity = np.loadtxt(
            opacity_file, skiprows=1, unpack=True)
    ind = np.where(np.logical_and(wavelengths.min() < wavelength_grid,
                                  wavelength_grid < wavelengths.max()))
    wavelength_grid, opacity = wavelength_grid[ind], opacity[ind]
    matisse_opacity = transform_spectrum_to_real_spectral_resolution(wavelength_grid, opacity,
                                                                     DL_COEFFS[resolution], 10,
                                                                     wavelengths,
                                                                     SPECTRAL_BINNING[resolution])
    if save_path is not None:
        np.save(save_path, [wavelengths, matisse_opacity])
    return matisse_opacity


def linearly_combine_opacities(weights: List[float], files: List[Path],
                               wavelength_solution: u.m) -> np.ndarray:
    """Linearly combines multiple opacities by their weights."""
    total_opacity = None
    for weight, file in zip(weights, files):
        opacity = opacity_to_matisse_opacity(wavelength_solution,
                                             qval_file=file)
        if total_opacity is None:
            total_opacity = weight*opacity
        else:
            total_opacity += weight*opacity
    return total_opacity
