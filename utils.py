from typing import Optional, Union

import astropy.units as u
import numpy as np


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
def calculate_intensity(wavelengths: u.um,
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
    spectral_profile = []
    pixel_size *= u.rad
    for wavelength in wavelengths*u.m:
        spectral_radiance = plancks_law(wavelength).to(
        u.erg/(u.cm**2*u.Hz*u.s*u.rad**2))
        spectral_profile.append((spectral_radiance*pixel_size**2).to(u.Jy).value)
    return np.array(spectral_profile)


def pad_image(image: np.ndarray):
    """Pads an image with additional zeros for Fourier transform."""
    im0 = np.sum(image, axis=(0, 1))
    dimy = im0.shape[0]
    dimx = im0.shape[1]

    im0x = np.sum(im0, axis=1)
    im0y = np.sum(im0, axis=1)

    s0x = np.trim_zeros(im0x).size
    s0y = np.trim_zeros(im0y).size

    min_sizex = s0x*oimOptions["FTpaddingFactor"]
    min_sizey = s0y*oimOptions["FTpaddingFactor"]

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
    if len(image.shape) <= 4:
        shape = (image.shape[0], image.shape[1], *binned_shape)
    else:
        shape = binned_shape
    if rdim:
        return image.reshape(shape).mean(-1).mean(-2), new_dim
    return image.reshape(shape).mean(-1).mean(-2)


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
