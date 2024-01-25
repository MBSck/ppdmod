import time as time
from pathlib import Path
from typing import Any, Callable, Optional, Dict, Tuple, List

import astropy.units as u
import numpy as np
from astropy.convolution import Gaussian1DKernel, Box1DKernel, convolve
from numpy.polynomial.polynomial import polyval
from openpyxl import Workbook, load_workbook
from scipy.interpolate import interp1d
from scipy.special import j1

from .options import OPTIONS


def take_time_average(func, *args, nsteps=10):
    """Takes a time average of the code."""
    execution_times = []
    for _ in range(nsteps):
        time_st = time.perf_counter()
        return_val = func(*args)
        execution_times.append(time.perf_counter()-time_st)
    return return_val, np.array(execution_times).mean()


def execution_time(func):
    """Prints the execution time of the decorated function."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        print(f"Execution time: {time.time() - start_time:.6f} seconds")
        return result
    return wrapper


def make_workbook(file: Path, sheets: Dict[str, List[str]]) -> None:
    """Creates an (.xslx)-sheet with subsheets.

    Parameters
    ----------
    file : Path
    sheets : dict of list
        A dictionary having the sheet name as they key
        and a list of columns as the value.
    """
    file_existed = True
    if file.exists():
        workbook = load_workbook(file)
    else:
        workbook = Workbook()
        file_existed = False
    for index, (sheet, columns) in enumerate(sheets.items()):
        if sheet not in workbook.sheetnames:
            if index == 0 and not file_existed:
                worksheet = workbook.active
                worksheet.title = sheet
            else:
                worksheet = workbook.create_sheet(title=sheet)
        else:
            worksheet = workbook[sheet]
        worksheet.delete_rows(1, worksheet.max_row)
        for col_idx, column_name in enumerate(columns, start=1):
            cell = worksheet.cell(row=1, column=col_idx)
            cell.value = column_name
    workbook.save(file)


def get_closest_indices(
        values, array: np.ndarray,
        window: Optional[float] = None,
        atol: Optional[float] = 1e-2) -> float:
    """Gets the closest indices of values occurring in a numpy array."""
    array = array.value if isinstance(array, u.Quantity) else array
    values = values.value if isinstance(values, u.Quantity) else values
    window = window.value if isinstance(window, u.Quantity) else window
    if not isinstance(values, (list, tuple, np.ndarray)):
        values = [values]

    indices = {}
    for value in values:
        if window is not None:
            index = np.where(np.logical_and((value-window) < array,
                                            (value+window) > array))[0]
        else:
            index = np.where(array == value)[0]
        if index.size == 0:
            index = np.where(np.abs(array - value) <= atol)[0]
        indices[str(value)] = index.astype(int).squeeze()
    return {key: value for key, value in indices.items() if value.size != 0}


def uniform_disk(pixel_size: u.mas, dim: int,
                 diameter: Optional[u.Quantity[u.mas]] = None) -> u.one:
    """The brightness profile of a uniform disk.

    Parameters
    ----------
    pixel_size : astropy.units.mas
        The size of a pixel in the image.
    dim : float
        The image's dimension [px].
    diameter : astropy.units.mas, optional
        The uniform disk's diameter.

    Returns
    -------
    radial_profile : astropy.units.one
    """
    if diameter is not None:
        v = np.linspace(-0.5, 0.5, dim, endpoint=False)\
            * pixel_size.to(u.mas)*dim
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


def binary(dim: int, pixel_size: u.mas,
           flux1: u.Jy, flux2: u.Jy,
           position1: u.mas, position2: u.mas) -> u.Jy:
    """The image of a binary.

    Parameters
    ----------
    dim : float
        The image's dimension [px].
    pixel_size : astropy.units.mas
        The size of a pixel in the image.
    flux1 : astropy.units.Jy
        The main component's flux.
    flux2 : astropy.units.Jy
        The companion's flux.
    position1 : astropy.units.m
        The main component's (x, y)-coordinates.
    position2 : astropy.units.m
        The companion's (x, y)-coordinates.
    wavelength : astropy.units.um

    Returns
    -------
    image : astropy.units.Jy
    """
    v = np.linspace(-0.5, 0.5, dim, endpoint=False)\
        * pixel_size.to(u.mas)*dim
    image = np.zeros((dim, dim))*u.Jy
    position1 = np.array(list(map(lambda x: np.where(v == x), position1))).flatten().tolist()
    position2 = np.array(list(map(lambda x: np.where(v == x), position2))).flatten().tolist()
    image[position1[1]][position1[0]] = flux1
    image[position2[1]][position2[0]] = flux2
    return image


def binary_vis(flux1: u.Jy, flux2: u.Jy,
               ucoord: u.m, vcoord: u.m,
               position1: u.mas, position2: u.mas,
               wavelength: u.um) -> np.ndarray:
    """The complex visibility function of a binary star.

    Parameters
    ----------
    flux1 : astropy.units.Jy
        The main component's flux.
    flux2 : astropy.units.Jy
        The companion's flux.
    position1 : astropy.units.m
        The main component's (x, y)-coordinates.
    position2 : astropy.units.m
        The companion's (x, y)-coordinates.
    wavelength : astropy.units.um

    Returns
    -------
    complex_visibility_function : numpy.ndarray
    """
    ucoord, vcoord = map(lambda x: x/wavelength.to(u.m), [ucoord, vcoord])
    xy_and_uv = list(
        map(lambda x: ucoord*x.to(u.rad)[0]+vcoord*x.to(u.rad)[1],
            [position1, position2]))
    return (flux1.value*np.exp(2*np.pi*1j*xy_and_uv[0].value)
            + flux2.value*np.exp(2*np.pi*1j*xy_and_uv[1].value))


def qval_to_opacity(qval_file: Path) -> u.cm**2/u.g:
    """Reads a qval file, then calculates and returns the
    opacity.

    Parameters
    ----------
    qval_file : pathlib.Path

    Returns
    -------
    opacity : astropy.units.cm**2/u.g

    Notes
    -----
    The qval-files give the grain size in microns and the
    density in g/cm^3.
    """
    with open(qval_file, "r+", encoding="utf8") as file:
        _, grain_size, density = map(float, file.readline().strip().split())
    wavelength_grid, qval = np.loadtxt(qval_file, skiprows=1, unpack=True)
    return wavelength_grid*u.um, \
        3*qval/(4*(grain_size*u.um).to(u.cm)*(density*u.g/u.cm**3))


def transform_data(
        wavelength_grid: u.um, data: u.Quantity, wavelength_solution: u.um,
        dl_coeffs: Optional[u.Quantity[u.one]] = OPTIONS["spectrum.coefficients"]["low"],
        spectral_binning: Optional[float] = 7,
        kernel_width: Optional[float] = 10
        ) -> u.Quantity:
    """Transform a spectrum to the real wavlength grid of MATISSE.
    Function written by József Varga.

    Parameters
    ----------
    wavelength_grid : astropy.units.um
    data : astropy.units.Quantity
    wavelength_solution : astropy.units.um
    dl_coeffs : list of float
    kernel_width : float
        The kernel width [px].
    spectral_binning : float

    Returns
    -------
    transformed_data : astropy.units.Quantity
    """
    min_wl, max_wl = np.min(wavelength_grid), np.max(wavelength_grid)
    wavelength, wl_new = min_wl, [min_wl]
    while wavelength < max_wl:
        wavelength = wavelength\
            + u.um*polyval(wavelength.value, dl_coeffs)/kernel_width
        wl_new.append(wavelength)
    wl_new = u.Quantity(wl_new, unit=u.um)
    f_spec_new = interp1d(wavelength_grid, data,
                          kind="cubic", fill_value="extrapolate")
    spec_new = f_spec_new(wl_new)

    # NOTE: Convolve with Gaussian kernel
    kernel = Gaussian1DKernel(stddev=kernel_width /
                              (2.0*np.sqrt(2.0*np.log(2.0))))
    spec_new[0] = np.nanmedian(spec_new[0:int(kernel.dimension/2.0)])
    spec_new[-1] = np.nanmedian(spec_new[-1:-int(kernel.dimension/2.0)])
    spec_convolved = convolve(spec_new, kernel, boundary="extend")

    # NOTE: Interpolate the convolved spectrum to the input wavelength grid
    f_spec_new = interp1d(wl_new, spec_convolved,
                          kind="cubic", fill_value="extrapolate")
    spec_interp = f_spec_new(wavelength_solution)

    # NOTE: Apply spectral binning: Convolve with a top-hat kernel of size
    # spectral_binning
    if spectral_binning > 1:
        kernel = Box1DKernel(spectral_binning)
        spec_final = convolve(spec_interp, kernel, boundary="extend")
    else:
        spec_final = spec_interp
    return spec_final


def data_to_matisse_grid(wavelength_solution: u.um,
                         data: Optional[Any] = None,
                         wavelength_grid: Optional[u.Quantity[u.um]] = None,
                         data_file: Optional[Path] = None,
                         unit: Optional[u.Quantity] = None,
                         resolution: Optional[str] = "low",
                         load_func: Optional[Callable] = None,
                         skiprows: Optional[str] = 1,
                         comments: Optional[str] = "#",
                         save_path: Optional[Path] = None) -> np.ndarray:
    """Interpolates an opacity from its wavelength grid to the
    MATISSE wavelength grid.

    Parameters
    ----------
    wavelength_solution : astropy.units.um
        The MATISSE wavelength solution.
    data : numpy.ndarray
        The data.
    wavelength_grid : astropy.units.um
        The wavelength grid.
    data_file : pathlib.Path
        The data file.
    unit : astropy.units, optional
        The unit of the data.
    resolution : str
        The resolution of the data.
    method : str
        The method to use.

    Returns
    -------
    numpy.ndarray
    """
    if data_file is not None:
        if load_func is not None:
            wavelength_grid, data = load_func(data_file)
        else:
            wavelength_grid, data, *_ = np.loadtxt(
                data_file, skiprows=skiprows,
                comments=comments, unpack=True)

    wavelength_grid = u.Quantity(wavelength_grid, unit=u.um)
    ind = np.where(np.logical_and(
        (wavelength_solution.min()-1*u.um) < wavelength_grid,
        wavelength_grid < (wavelength_solution.max()+1*u.um)))
    wavelength_grid, data = wavelength_grid[ind], data[ind]
    matisse_data = transform_data(
        wavelength_grid, data, wavelength_solution,
        OPTIONS["spectrum.coefficients"][resolution],
        OPTIONS["spectrum.binning"],
        OPTIONS["spectrum.kernel_width"])

    if save_path is not None:
        np.save(save_path, [wavelength_solution, matisse_data])
    return u.Quantity(matisse_data, unit=unit)


def linearly_combine_data(files: List[Path], weights: u.one,
                          wavelength_solution: u.um,
                          unit: Optional[u.Quantity] = None,
                          load_func: Optional[Callable] = None,
                          resolution: Optional[str] = "low") -> np.ndarray:
    """Linearly combines multiple opacities by their weights.

    Parameters
    ----------
    weights : u.one
        The weights for the different opacity components.
    files : list of pathlib.Path
    wavelength_solution : u.um
        The MATISSE wavelength solution.
    """
    combined_data = None
    for weight, data_file in zip(weights, files):
        data = data_to_matisse_grid(
                wavelength_solution, load_func=load_func,
                data_file=data_file, resolution=resolution, unit=unit)
        if combined_data is None:
            combined_data = weight*data
        else:
            combined_data += weight*data
    return combined_data


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

    .. math:: d = \\delta*D

    where 'd' is the diameter of the object and 'D' is the distance from the
    observer to the object and ..math::`\\delta` is the angular diameter.
    """
    return (angular_diameter.to(u.rad).value*distance.to(u.m))


def distance_to_angular(diameter: u.mas, distance: u.pc) -> u.m:
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
    return ((diameter.to(u.m)/distance.to(u.m))*u.rad).to(u.mas)


# TODO: Make function that takes care of rotation so all occurences are equal.
def calculate_effective_baselines(
        ucoord: u.m, vcoord: u.m, axis_ratio: u.one,
        pos_angle: u.deg) -> Tuple[u.Quantity[u.m], u.Quantity[u.one]]:
    """Calculates the effective baselines from the projected baselines
    in mega lambda.

    Parameters
    ----------
    ucoord: astropy.units.m
        The u coordinate.
    vcoord: astropy.units.m
        The v coordinate.
    axis_ratio: astropy.units.one
        The axis ratio of the ellipse
    pos_angle: astropy.units.deg
        The positional angle of the object
    wavelength: astropy.units.um, optional
        The wavelength.

    Returns
    -------
    astropy.units.m or astropy.units.one
        Returns the projected baselines either in meter or
        mega lambda.
    """
    if not isinstance(ucoord, u.Quantity):
        ucoord, vcoord = map(lambda x: x*u.m, [ucoord, vcoord])
    axis_ratio = axis_ratio*u.one\
        if not isinstance(axis_ratio, u.Quantity) else axis_ratio
    pos_angle = pos_angle*u.deg\
        if not isinstance(pos_angle, u.Quantity) else pos_angle

    ucoord_eff = ucoord*np.cos(pos_angle) - vcoord*np.sin(pos_angle)
    vcoord_eff = (ucoord*np.sin(pos_angle) + vcoord*np.cos(pos_angle))*axis_ratio
    return np.hypot(ucoord_eff, vcoord_eff), \
        np.arctan2(ucoord_eff, vcoord_eff)


def restrict_phase(phase: np.ndarray):
    """Restricts the phase to [-180, 180] degrees."""
    restricted_phase = phase % 360
    restricted_phase[restricted_phase > 180] -= 360
    return restricted_phase
