import time as time
from pathlib import Path
from typing import Callable, Optional, Dict, Tuple, List

import astropy.units as u
import astropy.constants as const
import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling.models import BlackBody
from matplotlib.axes import Axes
from matplotlib.legend import Legend
from openpyxl import Workbook, load_workbook
from scipy.special import j1

from .options import OPTIONS


def take_time_average(func: Callable, *args,
                      nsteps: Optional[int] = 10) -> Tuple[Callable, float]:
    """Takes a time average of the code."""
    execution_times = []
    for _ in range(nsteps):
        time_st = time.perf_counter()
        return_val = func(*args)
        execution_times.append(time.perf_counter()-time_st)
    return return_val, np.array(execution_times).mean()


def execution_time(func: Callable) -> Callable:
    """Prints the execution time of the decorated function."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        print(f"Execution time: {time.time() - start_time:.6f} seconds")
        return result
    return wrapper


def get_indices(values, array: np.ndarray,
                window: Optional[float] = None) -> List[np.ndarray]:
    """Gets the indices of values occurring in a numpy array
    and returns it in a list corresponding to the input values.

    Parameters
    ----------
    values : float
        The values to find.
    array : numpy.ndarray
        The array to search in.
    window : float
        The window around the value to search in.
    """
    array = array.value if isinstance(array, u.Quantity) else array
    values = values.value if isinstance(values, u.Quantity) else values
    window = window.value if isinstance(window, u.Quantity) else window
    if not isinstance(values, (list, tuple, np.ndarray)):
        values = [values]

    indices = []
    for value in values:
        if window is not None:
            index = np.where(((value-window) < array) & ((value+window) > array))[0]
        else:
            index = np.where(array == value)[0]

        indices.append(index.astype(int).squeeze())
    return indices


def compute_photometric_slope(
        wavelengths: u.um, temperature: u.K) -> np.ndarray:
    """Computes the photometric slope of the data from
    the effective temperature of the star.

    Parameters
    ----------
    wavelengths : astropy.units.um
        The wavelengths of the data.
    temperature : astropy.units.K
        The effective temperature of the star.

    Returns
    -------
    photometric_slope : numpy.ndarray
    """
    temperature = u.Quantity(temperature, u.K)
    wavelengths = u.Quantity(wavelengths, u.um)
    nu = (const.c/wavelengths.to(u.m)).to(u.Hz)
    blackbody = BlackBody(temperature)

    delta_nu = (nu * 1e-5)
    bb_upper, bb_lower = map(lambda x: blackbody(x), (nu+delta_nu, nu-delta_nu))
    bb_diff = (bb_upper-bb_lower) / (2 * delta_nu)
    photometric_slope = (nu / blackbody(nu)) * bb_diff
    return photometric_slope.value


def compute_stellar_radius(luminosity: u.Lsun, temperature: u.K) -> u.Rsun:
    """Calculates the stellar radius from the luminosity and temperature."""
    luminosity, temperature = u.Quantity(luminosity, u.Lsun), u.Quantity(temperature, u.K)
    return np.sqrt(luminosity.to(u.W)/(4*np.pi*const.sigma_sb*temperature**4)).to(u.Rsun)


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


def compute_effective_baselines(
        ucoord: u.m, vcoord: u.m,
        inclination: Optional[u.Quantity[u.one]] = None,
        pos_angle: Optional[u.Quantity[u.deg]] = None,
        longest: Optional[bool] = False
        ) -> Tuple[u.Quantity[u.m], u.Quantity[u.one]]:
    """Calculates the effective baselines from the projected baselines
    in mega lambda.

    Parameters
    ----------
    ucoord: astropy.units.m
        The u coordinate.
    vcoord: astropy.units.m
        The v coordinate.
    inclination: astropy.units.one
        The inclinatin induced compression of the x-axis.
    pos_angle: astropy.units.deg
        The positional angle of the object
    longest : bool, optional
        If True, the longest baselines are returned.

    Returns
    -------
    baselines : astropy.units.m
        Returns the effective baselines.
    baselines_angles : astropy.units.rad
        Returns the effective baseline angles.
    """
    ucoord, vcoord = map(lambda x: u.Quantity(x, u.m), [ucoord, vcoord])
    if pos_angle is not None:
        pos_angle = u.Quantity(pos_angle, u.deg)
        inclination = u.Quantity(inclination, u.one)

        ucoord_eff = ucoord*np.cos(pos_angle) - vcoord*np.sin(pos_angle)
        vcoord_eff = ucoord*np.sin(pos_angle) + vcoord*np.cos(pos_angle)
    else:
        ucoord_eff, vcoord_eff = ucoord, vcoord

    if inclination is not None:
        ucoord_eff *= inclination

    baselines_eff = np.hypot(ucoord_eff, vcoord_eff)
    baseline_angles_eff = np.arctan2(vcoord_eff, ucoord_eff)

    if longest:
        indices = baselines_eff.argmax(0)
        iteration = np.arange(baselines_eff.shape[1])
        baselines_eff = baselines_eff[indices, iteration]
        baseline_angles_eff = baseline_angles_eff[indices, iteration]

    return baselines_eff.squeeze(), baseline_angles_eff.squeeze()


def binary(dim: int, pixel_size: u.mas,
           flux1: u.Jy, flux2: u.Jy,
           position1: u.mas, position2: u.mas) -> u.Jy:
    """The image of a binary.

    Parameters
    ----------
    dim : float
        The image's dimension (px).
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
    wavelength_grid, qval = np.loadtxt(qval_file, skiprows=1,
                                       unpack=True, usecols=(0, 1))
    return wavelength_grid*u.um, \
        3*qval/(4*(grain_size*u.um).to(u.cm)*(density*u.g/u.cm**3))


def restrict_wavelength(
        wavelength: np.ndarray, array: np.ndarray,
        wavelength_range: u.um) -> Tuple[np.ndarray, np.ndarray]:
    """Restricts the wavelength for an input range."""
    indices = (wavelength > wavelength_range[0])\
        & (wavelength < wavelength_range[1])
    return wavelength[indices], array[indices]


def restrict_phase(phase: np.ndarray) -> np.ndarray:
    """Restricts the phase to [-180, 180] degrees."""
    restricted_phase = phase % 360
    restricted_phase[restricted_phase > 180] -= 360
    return restricted_phase


def get_opacity(source_dir: Path,weights: np.ndarray,
                sizes: List[List[float]],
                names: List[str], method: str,
                wavelength_grid: Optional[np.ndarray] = None,
                fmaxs: Optional[List[float]] = None,
                individual: Optional[bool] = False) -> Tuple[np.ndarray]:
    """Gets the opacity from input parameters."""
    qval_dict = {"olivine": "Q_Am_Mgolivine_Jae_DHS",
                 "pyroxene": "Q_Am_Mgpyroxene_Dor_DHS",
                 "forsterite": "Q_Fo_Suto_DHS",
                 "enstatite": "Q_En_Jaeger_DHS",
                 "silica": "Q_Silica_MH_DHS"}

    grf_dict = {"olivine": "MgOlivine",
                "pyroxene": "MgPyroxene",
                "forsterite": "Forsterite",
                "enstatite": "Enstatite"}

    # TODO: Include averaging over the crystalline silicates for DHS 0.1
    files = []
    for index, (size, name) in enumerate(zip(sizes, names)):
        for s in size:
            if method == "qval":
                file_name = f"{qval_dict[name]}_f{fmaxs[index]:.1f}_rv{s:.1f}.dat"
            else:
                file_name = f"{grf_dict[name]}{s:.1f}.Combined.Kappa"

            files.append(source_dir / method / file_name)

    load_func = qval_to_opacity if method == "qval" else None
    wl, opacity = load_data(files, load_func=load_func)

    if individual:
        return wl, opacity

    opacity = linearly_combine_data(opacity, weights)
    if wavelength_grid is not None:
        return wavelength_grid, np.interp(wavelength_grid, wl[0], opacity)
    return wl, opacity


def load_data(files: List[Path],
              wavelength_range: Optional[u.Quantity[u.um]] = [0.5, 15],
              load_func: Optional[Callable] = None,
              comments: Optional[str] = "#",
              skiprows: Optional[int] = 1,
              ) -> Tuple[np.ndarray, np.ndarray]:
    """Loads data from a file.

    Can either be one or multiple files, but in case
    of multiple files they need to have the same structure
    and size (as they will be converted to numpy.ndarrays).

    Parameters
    ----------
    files : list of pathlib.Path
    wavelength_range : tuple of float, optional
    load_func : callable, optional
    comments : str, optional
    skiprows : str, optional

    Returns
    -------
    wavelength_grid : numpy.ndarray
    data : numpy.ndarray
    """
    files = files if isinstance(files, list) else [files]
    wavelength_grids, data = [], []
    for file in files:
        if load_func is not None:
            wavelengths, content = load_func(file)
        else:
            wavelengths, content = np.loadtxt(
                file, skiprows=skiprows, usecols=(0, 1),
                comments=comments, unpack=True)

        if isinstance(wavelengths, u.Quantity):
            wavelengths = wavelengths.value
            content = content.value

        if wavelength_range is not None:
            wavelengths, content = restrict_wavelength(
                    wavelengths, content, wavelength_range)

        wavelength_grids.append(wavelengths)
        data.append(content)

    if len(files) > 1:
        min_shape = min(map(lambda x: x.shape[0], data))
        wl_ind = np.where(map(lambda x: x.shape[0] == min_shape, wavelength_grids))
        min_wl = wavelength_grids[wl_ind[0][0]]
        for index, (wl, d) in enumerate(zip(wavelength_grids, data)):
            if wl.shape[0] == min_shape:
                continue

            wavelength_grids[index] = min_wl
            data[index] = np.interp(min_wl, wl, d)

    return tuple(map(lambda x: np.array(x).squeeze(),
                     (wavelength_grids, data)))


def linearly_combine_data(data: np.ndarray, weights: u.one) -> np.ndarray:
    """Linearly combines multiple opacities by their weights.

    Parameters
    ----------
    data : numpy.ndarray
        The data to be linearly combined.
    weights : u.one
        The weights for the different data.

    Returns
    -------
    numpy.ndarray
    """
    return np.sum(data*weights[:, np.newaxis], axis=0)


def set_axes_color(ax: Axes, background_color: str) -> None:
    """Sets all the axes' facecolor."""
    opposite_color = "white" if background_color == "black" else "black"
    ax.set_facecolor(background_color)
    ax.spines['bottom'].set_color(opposite_color)
    ax.spines['top'].set_color(opposite_color)
    ax.spines['right'].set_color(opposite_color)
    ax.spines['left'].set_color(opposite_color)
    ax.xaxis.label.set_color(opposite_color)
    ax.yaxis.label.set_color(opposite_color)
    ax.tick_params(axis='x', colors=opposite_color)
    ax.tick_params(axis='y', colors=opposite_color)


def set_legend_color(legend: Legend, background_color: str) -> None:
    """Sets the legend's facecolor."""
    opposite_color = "white" if background_color == "black" else "black"
    plt.setp(legend.get_texts(), color=opposite_color)
    legend.get_frame().set_facecolor(background_color)


# TODO: Remove the check for the ucoord here
def broadcast_baselines(
        wavelength: u.um, baselines: u.m,
        baseline_angles: u.rad, ucoord: u.m
    ) -> Tuple[u.Quantity[u.um], u.Quantity[1/u.rad], u.Quantity[u.rad]]:
    """Broadcasts the baselines to the correct shape depending on
    the input ucoord shape."""
    wavelength = wavelength[:, np.newaxis]
    if ucoord.shape[0] == 1:
        baselines = (baselines/wavelength.to(u.m)).value
        baselines = baselines[..., np.newaxis]
        baseline_angles = baseline_angles[np.newaxis, :, np.newaxis]
    else:
        wavelength = wavelength[..., np.newaxis]
        baselines = (baselines[np.newaxis, ...]/wavelength.to(u.m)).value
        baselines = baselines[..., np.newaxis]
        baseline_angles = baseline_angles[np.newaxis, ..., np.newaxis]
    baseline_angles = u.Quantity(baseline_angles, unit=u.rad)
    return wavelength, baselines/u.rad, baseline_angles


def compute_t3(vis: np.ndarray) -> np.ndarray:
    """Computes the closure phase from the visibility function."""
    vis = np.concatenate((vis[:, :2], np.conj(vis[:, 2:])), axis=1)
    return np.angle(np.prod(vis, axis=1), deg=True).astype(OPTIONS.data.dtype.real)


def compute_vis(vis: np.ndarray) -> np.ndarray:
    """Computes the visibilities from the visibility function."""
    return np.abs(vis).astype(OPTIONS.data.dtype.real)
