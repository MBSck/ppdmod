import time as time
from pathlib import Path
from typing import Callable, Optional, Dict, Tuple, List

import astropy.units as u
import astropy.constants as const
import numpy as np
from astropy.convolution import Gaussian1DKernel, convolve_fft
from astropy.io import fits
from astropy.modeling.models import BlackBody
from openpyxl import Workbook, load_workbook
from scipy.special import j1
from scipy.interpolate import interp1d

from .options import OPTIONS, SPECTRAL_RESOLUTIONS, RESOLUTION_GRIDS


def resample_and_convolve(
        wavelengths: np.ndarray,
        wavelengths_data: np.ndarray, data: np.ndarray,
        sampling_dim: Optional[int] = 8192,
        padding: Optional[float] = 1,
        constant_resolution: Optional[bool] = False) -> u.um:
    """This function resamples the wavelengths in such a way that the
    convolution can be done with a kernel of constant resolution.

    Parameters
    ----------
    wavelengths : astropy.units.um
        The wavelengths to be resampled to.
    wavelengths_data : astropy.units.um
        The wavelengths of the data.
    data : numpy.ndarray
        The data to be convolved and resampled.
    sampling_dim : int
        The dimension to resample the data with.
    padding : astropy.units.um
        The padding to add to the data (some spacing left and right)
        to avoid edge effects.

    Returns
    -------
    convolved_data : numpy.ndarray
    """
    # HACK: Only possible to sort due to alphabetic order
    unique_bands = np.sort(list(set(np.array(list(map(get_band, wavelengths))))))
    data_resolutions = np.array([np.array(OPTIONS.data.resolutions)
                                [np.where(np.array(OPTIONS.data.bands) == (band if band not in ["lband", "mband"] else "lmband"))[0][0]]
                                for band in unique_bands])
    mock_grids = [np.linspace(*limit, 200) for limit in map(get_band_limits, unique_bands)]

    if constant_resolution:
        sub_grids = [np.logspace(np.log10(wl_min - padding), np.log10(wl_max + padding), sampling_dim)
                    for wl_min, wl_max in map(get_band_limits, unique_bands)]
        sub_data = [np.interp(sub_grid, wavelengths_data, data) for sub_grid in sub_grids]

        fine_resolutions = np.array([sub_grid[0] / np.diff(sub_grid)[0] for sub_grid in sub_grids])
        kernels = list(map(lambda x: Gaussian1DKernel(stddev=x / np.sqrt(8 * np.log(2))),
                        fine_resolutions / data_resolutions))
        convolved_data = [convolve_fft(sub_data, kernel) for sub_data, kernel in zip(sub_data, kernels)]
        interpolated_data = np.concatenate([np.interp(mock_grid, sub_grid, conv_data)
                                            for mock_grid, sub_grid, conv_data in zip(mock_grids, sub_grids, convolved_data)])
    else:
        interpolated_data = []
        for band_index, band in enumerate(unique_bands):
            band_resolution = data_resolutions[band_index]
            res_grid_wls, res_grids = RESOLUTION_GRIDS[band][band_resolution]

            oversampled_grid = []
            for index, (res_grid_wl, res_grid) in enumerate(zip(res_grid_wls, res_grids)):
                if index == len(res_grid_wls) - 1:
                    continue

                oversampled_grid.append(np.linspace(res_grid_wl, res_grid_wls[index + 1], int(res_grid * 1e3), endpoint=False))

            oversampled_grid = np.concatenate(oversampled_grid)
            oversampled_data = np.interp(oversampled_grid, wavelengths_data, data)
            convolved_data = convolve_fft(oversampled_data, Gaussian1DKernel(stddev=1 / (band_resolution * np.sqrt(8 * np.log(2)))))
            interpolated_data.append(np.interp(mock_grids[band_index], oversampled_grid, convolved_data))

        interpolated_data = np.concatenate(interpolated_data)

    return np.concatenate(mock_grids), interpolated_data


def get_band_limits(band: str) -> Tuple:
    """Gets the limits of the respective band"""
    match band:
        case "hband":
            return 1.5, 1.9
        case "kband":
            return 1.8, 2.4
        case "lband":
            return 2.5, 4.0
        case "mband":
            return 4.0, 6.0
        case "nband":
            return 7.5, 15.0


def get_band(wavelength: u.um) -> str:
    """Gets the band of the (.fits)-file."""
    wavelength = wavelength.value if isinstance(wavelength, u.Quantity) \
        else wavelength
    wl_min, wl_max = wavelength.min(), wavelength.max()
    if wl_min > 1.5 and wl_max < 1.9:
        return "hband"
    if wl_min > 1.8 and wl_max < 2.4:
        return "kband"
    if wl_min > 2.5 and wl_max < 4.0:
        return "lband"
    if wl_min >= 4.0 and wl_max < 6:
        return "mband"
    if wl_min > 2.5 and wl_max < 6:
        return "lmband"
    if wl_min > 7.5 and wl_max < 15:
        return "nband"
    return "unknown"


def get_resolution(header: fits.Header, band: str) -> int:
    """Gets the resolution of the band from the header."""
    match band:
        case "hband":
            return 30
        case "kband":
            return 22
        case "lmband":
            return SPECTRAL_RESOLUTIONS["lmband"][header["HIERARCH ESO INS DIL ID"].lower()]
        case "nband":
            return SPECTRAL_RESOLUTIONS["nband"][header["HIERARCH ESO INS DIN ID"].lower()]


def smooth_interpolation(
        interpolation_points: np.ndarray,
        grid: np.ndarray, values: np.ndarray,
        kind: Optional[str] = None,
        fill_value: Optional[str] = None) -> np.ndarray:
    """Rebins the grid to a higher factor and then interpolates and averages
    to the original grid.

    Parameters
    ----------
    interpolation_points : numpy.ndarray
        The points to interpolate to.
    points : numpy.ndarray
        The points to interpolate from.
    values : numpy.ndarray
        The values to interpolate.
    """
    kind = OPTIONS.data.interpolation.kind if kind is None else kind
    fill_value = OPTIONS.data.interpolation.fill_value if fill_value is None else fill_value
    points = interpolation_points.flatten()
    windows = np.array([getattr(OPTIONS.data.binning, get_band(point)).value for point in points])
    interpolation_grid = (np.linspace(-1, 1, OPTIONS.data.interpolation.dim) * windows[:, np.newaxis]).T + points
    return np.interp(interpolation_grid, grid, values).mean(axis=0).reshape(interpolation_points.shape)


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
            index = np.where(((value - window) < array) & ((value + window) > array))[0]
        else:
            index = np.where(array == value)[0]
            if index.size == 0:
                if value < array[0] or value > array[-1]:
                    indices.append(index.astype(int).flatten())
                    continue

                index = np.where(array == min(array, key=lambda x: abs(x - value)))[0]

        indices.append(index.astype(int).flatten())

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
    return np.gradient(np.log(blackbody(nu).value), np.log(nu.value))


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


def distance_to_angular(diameter: u.au, distance: u.pc) -> u.mas:
    """Converts an angular diameter of an object at a certain distance
    from the observer from mas to meters.

    Parameters
    ----------
    diameter : astropy.units.au
        The diameter of an object.
    distance : astropy.units.pc
        The distance to the object.

    Returns
    -------
    diameter : astropy.units.mas
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
        longest: Optional[bool] = False,
        rzero: Optional[bool] = True
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
    rzero : bool, optional
        If True, the zero-frequency is returned as well.

    Returns
    -------
    baselines : astropy.units.m
        Returns the effective baselines.
    baselines_angles : astropy.units.rad
        Returns the effective baseline angles.
    """
    ucoord, vcoord = map(lambda x: u.Quantity(x, u.m), [ucoord, vcoord])
    if pos_angle is not None:
        pos_angle = u.Quantity(pos_angle, u.deg).to(u.rad)
        inclination = u.Quantity(inclination, u.one)

        ucoord_eff = ucoord*np.cos(pos_angle) - vcoord*np.sin(pos_angle)
        vcoord_eff = ucoord*np.sin(pos_angle) + vcoord*np.cos(pos_angle)
    else:
        ucoord_eff, vcoord_eff = ucoord, vcoord

    if inclination is not None:
        ucoord_eff *= inclination

    baselines_eff = np.hypot(ucoord_eff, vcoord_eff)
    baseline_angles_eff = np.arctan2(vcoord_eff, ucoord_eff)

    # HACK: For some reason all my phases are mirrored?
    # baseline_angles_eff -= (180*u.deg).to(u.rad)

    if longest:
        indices = baselines_eff.argmax(0)
        iteration = np.arange(baselines_eff.shape[1])
        baselines_eff = baselines_eff[indices, iteration]
        baseline_angles_eff = baseline_angles_eff[indices, iteration]

    if not rzero:
        if len(baselines_eff.shape) > 1:
            baselines_eff = baselines_eff[:, 1:]
            baseline_angles_eff = baseline_angles_eff[:, 1:]
        else:
            baselines_eff = baselines_eff[1:]
            baseline_angles_eff = baseline_angles_eff[1:]

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


def get_opacity(source_dir: Path, weights: np.ndarray,
                names: List[str], method: str,
                wavelength_grid: Optional[np.ndarray] = None,
                fmaxs: Optional[List[float]] = None,
                individual: Optional[bool] = False,
                **kwargs) -> Tuple[np.ndarray]:
    """Gets the opacity from input parameters."""
    qval_dict = {"olivine": "Q_Am_Mgolivine_Jae_DHS",
                 "pyroxene": "Q_Am_Mgpyroxene_Dor_DHS",
                 "forsterite": "Q_Fo_Suto_DHS",
                 "enstatite": "Q_En_Jaeger_DHS",
                 "silica": "Q_silica"}

    grf_dict = {"olivine": "MgOlivine",
                "pyroxene": "MgPyroxene",
                "forsterite": "Forsterite",
                "enstatite": "Enstatite",
                "silica": "Silica"}

    files = []
    for index, name in enumerate(names):
        for size in ["small", "large"]:
            if method == "qval":
                size = 0.1 if size == "small" else 1.5
                if fmaxs[index] is not None:
                    file_name = f"{qval_dict[name]}_f{fmaxs[index]:.1f}_rv{size:.1f}.dat"
                else:
                    file_name = f"{qval_dict[name]}_rv{s:.1f}.dat"
            elif method == "grf":
                size = 0.1 if size == "small" else 2
                file_name = f"{grf_dict[name]}{size:.1f}.Combined.Kappa"
            else:
                size = "Big" if size == "large" else size.title()
                file_name = f"{prefix}{name.title()}.kappa"

            files.append(source_dir / method / file_name)

    load_func = qval_to_opacity if method == "qval" else None
    wl, opacity = load_data(files, load_func=load_func, **kwargs)

    if individual:
        return wl, opacity

    opacity = linearly_combine_data(opacity, weights)
    if wavelength_grid is not None:
        return wavelength_grid, np.interp(wavelength_grid, wl[0], opacity)
    return wl, opacity


def load_data(files: List[Path],
              load_func: Optional[Callable] = None,
              comments: Optional[str] = "#",
              skiprows: Optional[int] = 1,
              usecols: Optional[Tuple[int, int]] = (0, 1),
              method: Optional[str] = "shortest",
              kind: Optional[str] = None,
              fill_value: Optional[str] = None,
              ) -> Tuple[np.ndarray, np.ndarray]:
    """Loads data from a file.

    Can either be one or multiple files, but in case
    of multiple files they need to have the same structure
    and size (as they will be converted to numpy.ndarrays).

    Parameters
    ----------
    files : list of pathlib.Path
        The files to load the data from.
    load_func : callable, optional
        The function to load the data with.
    comments : str, optional
        Comment identifier.
    skiprows : str, optional
        The rows to skip.
    usecols : tuple of int, optional
        The columns to use.
    method : str, optional
        The grid to interpolate/extrapolate the data on.
        Default is 'shortest'. Other option is 'longest' or
        'median'.
    kind : str, optional
        The interpolation kind.
        Default is 'cubic'.
    fill_value : str, optional
        If "extrapolate", the data is extrapolated.
        Default is None.

    Returns
    -------
    wavelength_grid : numpy.ndarray
    data : numpy.ndarray
    """
    kind = OPTIONS.data.interpolation.kind if kind is None else kind
    fill_value = OPTIONS.data.interpolation.fill_value if fill_value is None else fill_value

    files = files if isinstance(files, list) else [files]
    wavelength_grids, contents = [], []
    for file in files:
        if load_func is not None:
            wavelengths, content = load_func(file)
        else:
            wavelengths, content = np.loadtxt(
                file, skiprows=skiprows, usecols=usecols,
                comments=comments, unpack=True)

        if isinstance(wavelengths, u.Quantity):
            wavelengths = wavelengths.value
            content = content.value

        wavelength_grids.append(wavelengths)
        contents.append(content)

    sizes = [np.size(wl) for wl in wavelength_grids]
    if method == "longest":
        wavelength_grid = wavelength_grids[np.argmax(sizes)]
    elif method == "shortest":
        wavelength_grid = wavelength_grids[np.argmin(sizes)]
    else:
        wavelength_grid = wavelength_grids[np.median(sizes).astype(int) == wavelength_grids]

    data = []
    for wavelengths, content in zip(wavelength_grids, contents):
        if np.array_equal(wavelengths, wavelength_grid):
            data.append(content)
            continue

        data.append(interp1d(wavelengths, content, kind=kind, fill_value=fill_value)(wavelength_grid))

    return wavelength_grid.squeeze(), np.array(data).squeeze()


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
    if vis.size == 0:
        return np.array([])
    return np.angle(vis[:, 0] * vis[:, 1] * vis[:, 2].conj(), deg=True).astype(OPTIONS.data.dtype.real)


def compute_vis(vis: np.ndarray) -> np.ndarray:
    """Computes the visibilities from the visibility function."""
    return np.abs(vis).astype(OPTIONS.data.dtype.real)
