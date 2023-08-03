import time
from pathlib import Path
from typing import Optional, Union, Any, Tuple, List

import astropy.units as u
import numpy as np
from astropy.convolution import Gaussian1DKernel, Box1DKernel, convolve
from astropy.modeling import models
from numpy.polynomial.polynomial import polyval
from scipy.interpolate import interp1d
from scipy.special import j1

from .options import OPTIONS
from .readout import ReadoutFits


def execution_time(func):
    """Prints the execution time of the decorated function."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        print(f"Execution time: {time.time() - start_time:.6f} seconds")
        return result
    return wrapper


def set_tuple_from_args(*args: Tuple) -> Tuple[Any]:
    """Sets the arguments to a tuple if a list is provided
    as the first argument. Otherwise return the arguments.
    """
    if isinstance(args[0], list):
        return tuple(arg for arg in args[0])
    return args


def set_fit_wavelengths(*wavelengths: Optional[u.m]) -> None:
    """Sets the wavelengths to be fitted for as a global option.

    If called without parameters or recalled, it will clear the
    fit wavelengths.
    """
    OPTIONS["fit.wavelengths"] = []

    if not wavelengths:
        return

    wavelengths = set_tuple_from_args(*wavelengths)
    if not isinstance(wavelengths, u.Quantity):
        wavelengths *= u.m
    OPTIONS["fit.wavelengths"] = wavelengths.to(u.um)


def set_data(*fits_files: Optional[Union[List[Path], Path]]) -> None:
    """Sets the data as a global variable from the input files.

    If called without parameters or recalled, it will clear the data.
    """
    OPTIONS["data.readouts"] = []
    OPTIONS["data.correlated_flux"],\
        OPTIONS["data.correlated_flux_error"] = [], []
    OPTIONS["data.closure_phase"],\
        OPTIONS["data.closure_phase_error"] = [], []

    if not fits_files:
        return

    fits_files = set_tuple_from_args(*fits_files)
    if not isinstance(fits_files, list):
        fits_files = [fits_files]
    readouts = OPTIONS["data.readouts"] =\
        [ReadoutFits(file) for file in fits_files]
    if not OPTIONS["fit.wavelengths"]:
        raise ValueError("Fitting wavelengths must be specified!")

    wavelengths = OPTIONS["fit.wavelengths"]
    for readout in readouts:
        OPTIONS["data.correlated_flux"].append(
            readout.get_data_for_wavelength(wavelengths, "vis"))
        OPTIONS["data.correlated_flux_error"].append(
            readout.get_data_for_wavelength(wavelengths, "vis_err"))
        OPTIONS["data.closure_phase"].append(
            readout.get_data_for_wavelength(wavelengths, "t3phi"))
        OPTIONS["data.closure_phase_error"].append(
            readout.get_data_for_wavelength(wavelengths, "t3phi_err"))


# TODO: Set the linespace endpoint=False for the real model as well.
def uniform_disk(pixel_size: u.mas, dim: int,
                 diameter: Optional[u.mas] = None) -> u.one:
    """The brightness profile of a uniform disk.

    Parameters
    ----------
    diameter : astropy.units.mas
        The uniform disk's diameter.
    pixel_size : astropy.units.mas
        The size of a pixel in the image.
    dim : float
        The image's dimension [px].

    Returns
    -------
    radial_profile : astropy.units.one
    """
    if diameter is not None:
        v = np.linspace(-0.5, 0.5, dim, endpoint=False)*pixel_size.to(u.mas)*dim
        x_arr, y_arr = np.meshgrid(v, v)
        radius = np.hypot(x_arr, y_arr) < diameter/2
    else:
        radius = np.ones((dim, dim)).astype(bool)
        diameter = 1*u.mas
    return 4*u.one*radius/(np.pi*diameter.value**2)


def uniform_disk_vis(diameter: u.mas, ucoord: u.m,
                     vcoord: u.m, wavelength: u.um) -> np.ndarray:
    """Defines the complex visibility function of a uniform disk.

    Parameters
    ----------
    diameter : astropy.units.mas
        The uniform disk's diameter.
    ucoord : astropy.units.m
        The u-coordinates.
    vcoord : astropy.units.m
        The v-coordinates.
    wavelength : astropy.units.um
        The wavelength for the spatial frequencies' unit conversion.

    Returns
    -------
    complex_visibility_function : numpy.ndarray
    """
    rho = np.hypot(ucoord, vcoord)/wavelength.to(u.m)
    return 2*j1(np.pi*rho*diameter.to(u.rad).value)\
        / (np.pi*diameter.to(u.rad).value*rho)


def get_next_power_of_two(number: Union[int, float]) -> int:
    """Returns the next higher power of two for an integer or float input.

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


def get_binned_dimension(dim: int, binning_factor: int) -> int:
    """Gets the binned dimension from the original dimension
    and the binning factor."""
    return int(dim*2**-binning_factor)


def rebin_image(image: np.ndarray,
                binning_factor: Optional[int] = None) -> np.ndarray:
    """Bins a 2D-image down according to the binning factor.

    Parameters
    ----------
    image : numpy.ndarray
        The image to be rebinned.
    binning_factor : int, optional
        The binning factor. The default is 0

    Returns
    -------
    rebinned_image : numpy.ndarray
        The rebinned image.
    dimension : int, optional
        The new dimension of the image.
    """
    if binning_factor is None:
        return image
    new_dim = get_binned_dimension(image.shape[-1], binning_factor)
    binned_shape = (new_dim, int(image.shape[-1] / new_dim),
                    new_dim, int(image.shape[-1] / new_dim))
    image = image.reshape(binned_shape)
    if OPTIONS["model.output"] == "surface_brightness":
        return image.mean(-1).mean(1)
    return image.sum(-1).sum(1)


# TODO: Check if this function does what it should -> Compare to oimodeler.
def pad_image(image: np.ndarray, padding_factor: int) -> np.ndarray:
    """Pads an image with additional zeros to avoid
    artefacts of aperiodicity before a Fourier transform.

    Parameters
    ----------
    image : numpy.ndarray
    padding_factor : int

    Returns
    -------
    padded_image : numpy.ndarray
    """
    new_dim = image.shape[0]*2**padding_factor
    padding = (new_dim-image.shape[0])//2
    return np.pad(
        image, ((padding, padding), (padding, padding)),
        mode='constant', constant_values=0)


def qval_to_opacity(qval_file: Path) -> u.cm**2/u.g:
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
    return wavelenght_grid*u.um,\
        3*qval*u.one/(4*(grain_size*u.um).to(u.cm)*(density*u.g/u.cm**3))


def transform_opacity(
        wavelength_grid: u.um, opacity: u.cm**2/u.g,
        dl_coeffs: u.one, kernel_width: float,
        wavelength_solution: u.um, spectral_binning: float):
    """Transform a spectrum to the real wavlength grid of MATISSE.

    Parameters
    ----------
    wavelength_grid : astropy.units.um
    opacity : astropy.units.cm**2/astropy.units.g
    dl_coeffs : list of float
    kernel_width : float
        The kernel width [px].
    wavelength_solution : astropy.units.um
    spectral_binning : float

    Returns
    -------
    transformed_opacity : astropy.units.cm**2/astropy.units.g
    """
    min_wl, max_wl = np.min(wavelength_grid), np.max(wavelength_grid)
    wavelength, wl_new = min_wl, [min_wl]
    while wavelength < max_wl:
        wavelength = wavelength\
            + u.um*polyval(wavelength.value, dl_coeffs)/kernel_width
        wl_new.append(wavelength)
    wl_new = u.Quantity(wl_new, unit=u.um)
    f_spec_new = interp1d(wavelength_grid, opacity,
                          kind='cubic', fill_value='extrapolate')
    spec_new = f_spec_new(wl_new)

    # NOTE: Convolve with Gaussian kernel
    kernel = Gaussian1DKernel(stddev=kernel_width /
                              (2.0*np.sqrt(2.0*np.log(2.0))))
    spec_new[0] = np.nanmedian(spec_new[0:int(kernel.dimension/2.0)])
    spec_new[-1] = np.nanmedian(spec_new[-1:-int(kernel.dimension/2.0)])
    spec_convolved = convolve(spec_new, kernel, boundary='extend')

    # NOTE: Interpolate the convolved spectrum to the input wavelength grid
    f_spec_new = interp1d(wl_new, spec_convolved,
                          kind='cubic', fill_value='extrapolate')
    spec_interp = f_spec_new(wavelength_solution)

    # NOTE: Apply spectral binning: Convolve with a top-hat kernel of size
    # spectral_binning
    if spectral_binning > 1:
        kernel = Box1DKernel(spectral_binning)
        spec_final = convolve(spec_interp, kernel, boundary='extend')
    else:
        spec_final = spec_interp
    return spec_final*u.cm**2/u.g


def opacity_to_matisse_opacity(wavelength_solution: u.um,
                               opacity: Optional[np.ndarray] = None,
                               opacity_file: Optional[Path] = None,
                               qval_file: Optional[Path] = None,
                               resolution: Optional[str] = "low",
                               save_path: Optional[Path] = None) -> np.ndarray:
    """Converts the opacity to the MATISSE wavelength grid.

    Parameters
    ----------
    wavelength_solution : u.um
        The MATISSE wavelength solution.
    opacity : , optional
        An input opacity.
    opacity_file : pathlib.Path, optional
    qval_file : pathlib.Path, optional
    resolution : str, optional
    save_path : pathlib.Path, optional

    Returns
    -------
    numpy.ndarray
    """
    if qval_file is not None:
        wavelength_grid, opacity = qval_to_opacity(qval_file)
    elif opacity_file is not None:
        wavelength_grid, opacity = np.loadtxt(
            opacity_file, skiprows=1, unpack=True)

    ind = np.where(np.logical_and(wavelength_solution.min() < wavelength_grid,
                                  wavelength_grid < wavelength_solution.max()))
    wavelength_grid, opacity = wavelength_grid[ind], opacity[ind]
    matisse_opacity = transform_opacity(
        wavelength_grid, opacity,
        OPTIONS["spectrum.coefficients"][resolution],
        10, wavelength_solution,
        OPTIONS["spectrum.binning"][resolution])
    if save_path is not None:
        np.save(save_path, [wavelength_solution, matisse_opacity])
    return matisse_opacity


def linearly_combine_opacities(weights: u.one, files: List[Path],
                               wavelength_solution: u.um) -> np.ndarray:
    """Linearly combines multiple opacities by their weights.

    Parameters
    ----------
    weights : u.one
        The weights for the different opacity components.
    files : list of pathlib.Path
    wavelength_solution : u.um
        The MATISSE wavelength solution.
    """
    total_opacity = None
    for weight, file in zip(weights, files):
        opacity = opacity_to_matisse_opacity(wavelength_solution,
                                             qval_file=file)
        if total_opacity is None:
            total_opacity = weight*opacity
        else:
            total_opacity += weight*opacity
    return total_opacity


def angular_to_distance(angular_diameter: u.mas, distance: u.pc) -> u.m:
    """Converts an angular diameter of an object at a certain distance
    from the observer from mas to meters.

    Parameters
    ----------
    angular_diameter : astropy.units.mas
        The angular diameter of an object.
    distance : astropy.units.pc
        The distance to the object.

    Returns
    -------
    diameter : astropy.units.m
        The diameter of the object.

    Notes
    -----
    The formula for the angular diameter small angle approximation is

    .. math:: \\delta = \\frac{d}{D}

    where 'd' is the diameter of the object and 'D' is the distance from the
    observer to the object and ..math::`\\delta` is the angular diameter.
    """
    return (angular_diameter.to(u.rad).value*distance.to(u.au)).to(u.m)


# TODO: Maybe even optimize calculation time further in the future
# Got it down from 3.5s to 0.66s for 3 wavelengths. It is 0.19s per wl.
def calculate_intensity(temp_profile: u.K,
                        wavelength: u.um,
                        pixel_size: Optional[u.rad] = None) -> np.ndarray:
    """Calculates the blackbody_profile via Planck's law and the
    emissivity_factor for a given wavelength, temperature- and
    dust surface density profile.

    Parameters
    ----------
    wavelengths : astropy.units.um
        Wavelength value(s).
    temp_profile : astropy.units.K
        Temperature profile.
    pixel_size : astropy.units.rad, optional
        The pixel size.

    Returns
    -------
    intensity : astropy.units.Jy
        Intensity per pixel [Jy/px]
    """
    plancks_law = models.BlackBody(temperature=temp_profile)
    spectral_radiance = plancks_law(wavelength.to(u.m)).to(
        u.erg/(u.cm**2*u.Hz*u.s*u.rad**2))
    return (spectral_radiance*(pixel_size.to(u.rad))**2).to(u.Jy)
