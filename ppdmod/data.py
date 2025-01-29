from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import astropy.units as u
import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.stats import circmean, circstd

from .options import OPTIONS
from .utils import get_band, get_band_limits, get_indices, get_binning_windows


class ReadoutFits:
    """All functionality to work with (.fits) or flux files.

    Parameters
    ----------
    fits_file : pathlib.Path
        The path to the (.fits) or flux file.
    """

    def __init__(self, fits_file: Path) -> None:
        """The class's constructor."""
        self.fits_file = Path(fits_file)
        self.name = self.fits_file.name
        self.band = "unknown"
        self.read_file()

    def read_file(self) -> None:
        """Reads the data of the (.fits)-files into vectors."""
        with fits.open(self.fits_file) as hdul:
            instrument = None
            if "instrume" in hdul[0].header:
                instrument = hdul[0].header["instrume"].lower()
            sci_index = OPTIONS.data.gravity.index if instrument == "gravity" else None
            self.wavelength = (
                hdul["oi_wavelength", sci_index].data["eff_wave"] * u.m
            ).to(u.um)
            self.array = "ats" if "AT" in hdul["oi_array"].data["tel_name"][0] else "uts"
            self.band = get_band(self.wavelength)
            self.flux = self.read_into_namespace(hdul, "flux", sci_index)
            self.t3 = self.read_into_namespace(hdul, "t3", sci_index)
            self.vis = self.read_into_namespace(hdul, "vis", sci_index)
            self.vis2 = self.read_into_namespace(hdul, "vis2", sci_index)

    def read_into_namespace(
        self,
        hdul: fits.HDUList,
        key: str,
        sci_index: int | None = None,
    ) -> SimpleNamespace:
        """Reads a (.fits) card into a namespace."""
        try:
            data = hdul[f"oi_{key}", sci_index].data
        except KeyError:
            return SimpleNamespace(
                value=np.array([]),
                err=np.array([]),
                ucoord=np.array([]).reshape(1, -1),
                vcoord=np.array([]).reshape(1, -1),
            )

        if key == "flux":
            try:
                return SimpleNamespace(
                    value=np.ma.masked_array(data["fluxdata"], mask=data["flag"]),
                    err=np.ma.masked_array(data["fluxerr"], mask=data["flag"]),
                )
            except KeyError:
                return SimpleNamespace(value=np.array([]), err=np.array([]), flag=np.array([]))

        # TODO: Might err if vis is not included in datasets
        if key in ["vis", "vis2"]:
            if key == "vis":
                value_key, err_key = "visamp", "visamperr"
            else:
                value_key, err_key = "vis2data", "vis2err"

            ucoord = data["ucoord"].reshape(1, -1).astype(OPTIONS.data.dtype.real)
            vcoord = data["vcoord"].reshape(1, -1).astype(OPTIONS.data.dtype.real)
            return SimpleNamespace(
                value=np.ma.masked_array(data[value_key], mask=data["flag"]),
                err=np.ma.masked_array(data[err_key], mask=data["flag"]),
                ucoord=np.round(ucoord, 2),
                vcoord=np.round(vcoord, 2),
            )

        u1coord, u2coord = map(lambda x: data[f"u{x}coord"], ["1", "2"])
        v1coord, v2coord = map(lambda x: data[f"v{x}coord"], ["1", "2"])
        u3coord, v3coord = u1coord + u2coord, v1coord + v2coord
        u123coord = np.array([u1coord, u2coord, u3coord]).astype(OPTIONS.data.dtype.real)
        v123coord = np.array([v1coord, v2coord, v3coord]).astype(OPTIONS.data.dtype.real)
        return SimpleNamespace(
            value=np.ma.masked_array(data["t3phi"], mask=data["flag"]),
            err=np.ma.masked_array(data["t3phierr"], mask=data["flag"]),
            u123coord=np.round(u123coord, 2),
            v123coord=np.round(v123coord, 2),
        )

    def get_data_for_wavelength(
        self,
        wavelength: u.Quantity,
        key: str,
        do_bin: bool = True,
    ) -> np.ma.masked_array:
        """Gets the data for the given wavelengths.

        If there is no data for the given wavelengths,
        a np.nan array is returned of the shape
        (wavelength.size, data.shape[0]).

        Parameters
        ----------
        wavelength : astropy.units.um
            The wavelengths to be returned.
        key : str
            The key (header) of the data to be returned.
        do_bin : bool, optional
            If the data should be binned or not.

        Returns
        -------
        numpy.ma.masked_array
            The data for the given wavelengths.
        """
        windows = get_binning_windows(wavelength)
        indices = get_indices(wavelength, array=self.wavelength, windows=windows)
        value, err = getattr(self, key).value, getattr(self, key).err
        if all(index.size == 0 for index in indices):
            nan_value = np.full((wavelength.size, value.shape[0]), np.nan)
            nan_err = np.full((wavelength.size, err.shape[0]), np.nan)
            if "flux":
                wl_value, wl_err = nan_value, nan_err
            else:
                wl_value, wl_err = nan_value[:, :1], nan_err[:, :1]

            wl_value = np.ma.masked_invalid(wl_value)
            wl_err = np.ma.masked_invalid(wl_err)
            return wl_value, wl_err

        if key == "t3":
            # TODO: Make sure this works properly with the masked arrays
            mean_func = partial(circmean, low=-180, high=180)
            std_func = partial(circstd, low=-180, high=180)
        else:
            mean_func, std_func = np.ma.mean, np.ma.std

        # TODO: This could be rewritten and shortened
        if do_bin:
            if key == "flux":
                wl_value = [np.ma.masked_array([mean_func(value.flatten()[index])]) for index in indices]
                wl_err = [np.ma.masked_array([np.ma.sqrt(np.ma.sum(err.flatten()[index] ** 2) + std_func(err.flatten()[index]) ** 2)]) for index in indices]
            else:
                wl_value = [mean_func(value[:, index], axis=-1) for index in indices]
                wl_err = [np.ma.sqrt(np.ma.sum(err[:, index] ** 2, axis=-1) + std_func(err[:, index], axis=-1) ** 2) for index in indices]
        else:
            wl_value = [
                value[:, index].flatten()
                if index.size != 0
                else np.ma.masked_invalid(np.full((value.shape[0],), np.nan))
                for index in indices
            ]
            wl_err = [
                err[:, index].flatten()
                if index.size != 0
                else np.ma.masked_invalid(np.full((err.shape[0],), np.nan))
                for index in indices
            ]

        wl_value = np.ma.masked_array(wl_value).astype(OPTIONS.data.dtype.real)
        wl_err = np.ma.masked_array(wl_err).astype(OPTIONS.data.dtype.real)
        return wl_value, wl_err


def get_all_wavelengths(readouts: List[ReadoutFits] | None = None) -> np.ndarray:
    """Gets all wavelengths from the readouts."""
    readouts = OPTIONS.data.readouts if readouts is None else readouts
    wavelengths = list(map(lambda x: x.wavelength, readouts))
    return np.sort(np.unique(np.concatenate(wavelengths)))


def set_fit_wavelengths(
    wavelengths: u.Quantity[u.um] | None = None,
) -> str | np.ndarray:
    """Sets the wavelengths to be fitted for as a global option.

    If called without a wavelength and all set to False, it will clear
    the fit wavelengths.

    Parameters
    ----------
    wavelengts : numpy.ndarray
        The wavelengths to be fitted.

    Returns
    -------
    str or numpy.ndarray
        The wavelengths to be fitted as a numpy array or "all" if all are to be
        fitted.
    """
    OPTIONS.fit.wavelengths = None
    if wavelengths is None:
        return

    wavelengths = u.Quantity(wavelengths, u.um)
    if wavelengths.shape == ():
        wavelengths = wavelengths.reshape((wavelengths.size,))
    OPTIONS.fit.wavelengths = wavelengths.flatten()
    return OPTIONS.fit.wavelengths


def get_counts_data() -> np.ndarray:
    """Gets the number of data points for the flux,
    visibilities and closure phases."""
    counts = []
    for key in OPTIONS.fit.data:
        data = getattr(OPTIONS.data, key)
        counts.append(data.value.compressed().size)

    return np.array(counts)


def clear_data() -> List[str]:
    """Clears data and returns the keys of the cleared data."""
    OPTIONS.fit.wavelengths = None
    OPTIONS.data.readouts = []

    for key in ["flux", "vis", "vis2", "t3"]:
        data = getattr(OPTIONS.data, key)
        data.value, data.err = [np.array([]) for _ in range(2)]
        if key in ["vis", "vis2"]:
            data.ucoord, data.vcoord = [np.array([]).reshape(1, -1) for _ in range(2)]
        elif key in "t3":
            data.u123coord, data.v123coord = [np.array([]) for _ in range(2)]

    return ["flux", "vis", "vis2", "t3"]


def read_data(data_to_read: List[str], wavelengths: u.um, min_err: float) -> None:
    """Reads in the data from the keys."""
    OPTIONS.data.nbaselines = []
    for readout in OPTIONS.data.readouts:
        for key in data_to_read:
            data = getattr(OPTIONS.data, key)
            data_readout = getattr(readout, key)
            value, err = readout.get_data_for_wavelength(
                wavelengths, key, OPTIONS.data.do_bin
            )
            if key in ["vis", "vis2", "t3"]:
                ind = np.where(np.ma.abs(err / value) < min_err)
                err[ind] = np.ma.abs(value[ind]) * min_err

            if data.value.size == 0:
                data.value, data.err = value, err
            else:
                data.value = np.ma.hstack((data.value, value))
                data.err = np.ma.hstack((data.err, err))

            if key in ["vis", "vis2"]:
                if data.ucoord.size == 0:
                    data.ucoord = np.insert(data_readout.ucoord, 0, 0, axis=1)
                    data.vcoord = np.insert(data_readout.vcoord, 0, 0, axis=1)
                else:
                    data.ucoord = np.concatenate(
                        (data.ucoord, data_readout.ucoord), axis=-1
                    )
                    data.vcoord = np.concatenate(
                        (data.vcoord, data_readout.vcoord), axis=-1
                    )

                # TODO: Make sure to avoid error if vis doesn't exists in OIFITS file
                if key == "vis2":
                    OPTIONS.data.nbaselines.append(data_readout.ucoord.size)

            elif key == "t3":
                uvcoords = np.stack((data_readout.u123coord, data_readout.v123coord), axis=-1)
                unique_uvcoords = np.unique(uvcoords.reshape(-1, 2), axis=0)
                ucoord, vcoord = unique_uvcoords[:, 0], unique_uvcoords[:, 1]
                index123 = np.vectorize(lambda x: np.where(ucoord == x)[0][0])(data_readout.u123coord)
                if data.u123coord.size == 0:
                    data.u123coord = np.insert(data_readout.u123coord, 0, 0, axis=1)
                    data.v123coord = np.insert(data_readout.v123coord, 0, 0, axis=1)
                    data.ucoord = np.insert(ucoord, 0, 0).reshape(1, -1)
                    data.vcoord = np.insert(vcoord, 0, 0).reshape(1, -1)
                    data.index123 = index123 + 1
                else:
                    data.u123coord = np.hstack((data.u123coord, data_readout.u123coord))
                    data.v123coord = np.hstack((data.v123coord, data_readout.v123coord))
                    data.ucoord = np.concatenate((data.ucoord, ucoord.reshape(1, -1)), axis=-1)
                    data.vcoord = np.concatenate((data.vcoord, vcoord.reshape(1, -1)), axis=-1)
                    data.index123 = np.hstack((data.index123, index123 + data.index123.max() + 1))

    for key in data_to_read:
        data = getattr(OPTIONS.data, key)
        data.value = np.ma.masked_invalid(data.value)
        data.err = np.ma.masked_invalid(data.err)


def average_data() -> None:
    """Averages the flux data and applys a correction factor to the correlated flux."""
    wavelengths = OPTIONS.fit.wavelengths
    flux, flux_err = OPTIONS.data.flux.value, OPTIONS.data.flux.err
    flux_averaged = np.ma.average(flux, weights=1 / flux_err**2, axis=-1)
    flux_err_averaged = np.ma.sqrt(1 / np.ma.sum(flux_err, axis=-1) ** 2)
    ind = np.where(flux_err_averaged < flux_averaged * 0.05)
    flux_err_averaged[ind] = flux_averaged[ind] * 0.05
    flux_ratio = flux / flux_averaged[:, np.newaxis]

    # NOTE: This sets the flux_ratio to 1 for files without any flux in a certain band
    # and otherwise interpolates any missing flux ratios.
    for index, band in enumerate(OPTIONS.data.bands):
        if band == "lmband":
            limits_lband = get_band_limits("lband") * u.um
            limits_mband = get_band_limits("mband") * u.um
            cond_lband = (limits_lband[0] < wavelengths) & (limits_lband[1] > wavelengths)
            cond_mband = (limits_mband[0] < wavelengths) & (limits_mband[1] > wavelengths)
            ind = np.where(cond_lband | cond_mband)[0]
        else:
            limits = get_band_limits(band) * u.um
            ind = np.where((limits[0] < wavelengths) & (limits[1] > wavelengths))[0]

        band_ratio = flux_ratio[:, index][ind]
        if band_ratio.compressed().size == band_ratio.size:
            continue

        if band_ratio.compressed().size == 0:
            flux_ratio[:, index][ind] = 1.
        elif band_ratio.compressed().size == 1:
            flux_ratio[:, index][ind] = band_ratio.compressed()
        else:
            interp_ratios = interp1d(wavelengths[ind][~band_ratio.mask],
                                    band_ratio.compressed(), fill_value="extrapolate")
            flux_ratio[:, index][ind] = interp_ratios(wavelengths[ind])


    for key in ["vis", "vis2"]:
        value = getattr(OPTIONS.data, key).value
        split_indices = np.cumsum(OPTIONS.data.nbaselines[:-1])
        for index, current_slice in enumerate(split_indices):
            prev_slice = None if index == 0 else split_indices[index - 1]
            current_slice = current_slice if index != len(split_indices) - 1 else None
            value[:, prev_slice:current_slice] = (
                value[:, prev_slice:current_slice] * flux_ratio[:, index][:, np.newaxis]
            )

    OPTIONS.data.flux.value = flux_averaged[:, np.newaxis]
    OPTIONS.data.flux.err = flux_err_averaged[:, np.newaxis]


def correct_data_errors(
    data_to_read: List[str], wavelengths: u.um, set_std_err: List[str], min_err: float
) -> None:
    """Corrects the errors for the data."""
    for key in data_to_read:
        data = getattr(OPTIONS.data, key)
        bands = np.array(list(map(get_band, wavelengths)))
        for band in set_std_err:
            ind = np.where(band == bands)[0]
            band_data = data.value[ind, :]
            band_std = np.tile(np.std(band_data, axis=0), (band_data.shape[0], 1))
            err_ind = np.where(np.abs(band_std / band_data) < min_err)
            band_std[err_ind] = np.abs(band_data[err_ind]) * min_err
            data.err[ind, :] = band_std


def set_data(
    fits_files: List[Path] | None = None,
    wavelengths: str | u.Quantity[u.um] | None = None,
    fit_data: List[str] = ["flux", "vis", "t3"],
    weights: Dict[str, float] | None = None,
    set_std_err: List[str] | None = None,
    min_err: float = 0.05,
    filter_by_array: str | None = None,
    average: bool = False,
    **kwargs,
) -> SimpleNamespace:
    """Sets the data as a global variable from the input files.

    If called without parameters or recalled, it will clear the data.

    Parameters
    ----------
    fits_files : list of Path
        The paths to the MATISSE (.fits)-files.
    wavelengts : str or numpy.ndarray
        The wavelengths to be fitted as a numpy array or "all" if all are to be
        fitted.
    fit_data : list of str, optional
        The data to be fitted.
    weights : list of float, optional
        The weights of the interferometric datasets used for fitting.
    set_std_err : list of str, optional
        The data to be set the standard error from the variance of the datasets from.
    min_err : float, optional
        The minimum error of the data to be kept. Will set the error to be that at least.
    average : bool, optional
        If toggled will average the flux over all files and set an offset for the correlated fluxes/visibilities.
    """
    data_to_read = clear_data()
    if fits_files is None:
        return OPTIONS.data

    OPTIONS.fit.data = fit_data
    readouts = OPTIONS.data.readouts = list(map(ReadoutFits, fits_files))
    if filter_by_array is not None:
        OPTIONS.data.readouts = list(filter(lambda x: x.array == filter_by_array, readouts))

    OPTIONS.data.bands = list(map(lambda x: x.band, OPTIONS.data.readouts))

    if wavelengths == "all":
        wavelengths = get_all_wavelengths(OPTIONS.data.readouts)
        OPTIONS.data.do_bin = True

    if wavelengths is None:
        raise ValueError("No wavelengths given and/or not 'all' specified!")

    wavelengths = set_fit_wavelengths(wavelengths)
    read_data(data_to_read, wavelengths, min_err)

    # TODO: Averaging does not work -> Do not use as of right now
    if average:
        average_data()

    if set_std_err is not None:
        correct_data_errors(data_to_read, wavelengths, set_std_err, min_err)

    if weights is not None:
        OPTIONS.fit.weights = SimpleNamespace(**weights)

    return OPTIONS.data
