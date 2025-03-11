import copy
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
from .utils import get_band, get_band_limits, get_binning_windows, get_indices


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
            self.wl = (hdul["oi_wavelength", sci_index].data["eff_wave"] * u.m).to(u.um)
            self.band = get_band(self.wl)
            self.array = (
                "ats" if "AT" in hdul["oi_array"].data["tel_name"][0] else "uts"
            )
            self.flux = self.read_into_namespace(hdul, "flux", sci_index)
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
            hdu = hdul[f"oi_{key}", sci_index]
            data = hdu.data
        except KeyError:
            return SimpleNamespace(
                val=np.array([]),
                err=np.array([]),
                u=np.array([]).reshape(1, -1),
                v=np.array([]).reshape(1, -1),
            )

        if key == "flux":
            try:
                return SimpleNamespace(
                    val=np.ma.masked_array(data["fluxdata"], mask=data["flag"]),
                    err=np.ma.masked_array(data["fluxerr"], mask=data["flag"]),
                )
            except KeyError:
                return SimpleNamespace(
                    val=np.array([]), err=np.array([]), flag=np.array([])
                )

        # TODO: Might err if vis is not included in datasets
        if key in ["vis", "vis2"]:
            if key == "vis":
                val_key, err_key = "visamp", "visamperr"
            else:
                val_key, err_key = "vis2data", "vis2err"

            ucoord = data["ucoord"].reshape(1, -1).astype(OPTIONS.data.dtype.real)
            vcoord = data["vcoord"].reshape(1, -1).astype(OPTIONS.data.dtype.real)
            return SimpleNamespace(
                val=np.ma.masked_array(data[val_key], mask=data["flag"]),
                err=np.ma.masked_array(data[err_key], mask=data["flag"]),
                u=np.round(ucoord, 2),
                v=np.round(vcoord, 2),
            )

        u1, u2 = map(lambda x: data[f"u{x}coord"], ["1", "2"])
        v1, v2 = map(lambda x: data[f"v{x}coord"], ["1", "2"])
        u123 = np.array([u1, u2, u1 + u2]).astype(OPTIONS.data.dtype.real)
        v123 = np.array([v1, v2, v1 + v2]).astype(OPTIONS.data.dtype.real)
        return SimpleNamespace(
            val=np.ma.masked_array(data["t3phi"], mask=data["flag"]),
            err=np.ma.masked_array(data["t3phierr"], mask=data["flag"]),
            u123=np.round(u123, 2),
            v123=np.round(v123, 2),
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
        indices = get_indices(wavelength, array=self.wl, windows=windows)
        val, err = getattr(self, key).val, getattr(self, key).err
        if all(index.size == 0 for index in indices):
            nan_val = np.full((wavelength.size, val.shape[0]), np.nan)
            nan_err = np.full((wavelength.size, err.shape[0]), np.nan)
            if "flux":
                wl_val, wl_err = nan_val, nan_err
            else:
                wl_val, wl_err = nan_val[:, :1], nan_err[:, :1]

            return np.ma.masked_invalid(wl_val), np.ma.masked_invalid(wl_err)

        if key == "t3":
            # TODO: Make sure this works properly with the masked arrays
            mean_func = partial(circmean, low=-180, high=180)
            std_func = partial(circstd, low=-180, high=180)
        else:
            mean_func, std_func = np.ma.mean, np.ma.std

        # TODO: This could be rewritten and shortened
        if do_bin:
            if key == "flux":
                wl_val = [
                    np.ma.masked_array([mean_func(val.flatten()[index])])
                    for index in indices
                ]
                wl_err = [
                    np.ma.masked_array(
                        [
                            np.ma.sqrt(
                                np.ma.sum(err.flatten()[index] ** 2)
                                + std_func(err.flatten()[index]) ** 2
                            )
                        ]
                    )
                    for index in indices
                ]
            else:
                wl_val = [mean_func(val[:, index], axis=-1) for index in indices]
                wl_err = [
                    np.ma.sqrt(
                        np.ma.sum(err[:, index] ** 2, axis=-1)
                        + std_func(err[:, index], axis=-1) ** 2
                    )
                    for index in indices
                ]
        else:
            wl_val = [
                (
                    val[:, index].flatten()
                    if index.size != 0
                    else np.ma.masked_invalid(np.full((val.shape[0],), np.nan))
                )
                for index in indices
            ]
            wl_err = [
                (
                    err[:, index].flatten()
                    if index.size != 0
                    else np.ma.masked_invalid(np.full((err.shape[0],), np.nan))
                )
                for index in indices
            ]

        wl_val = np.ma.masked_array(wl_val).astype(OPTIONS.data.dtype.real)
        wl_err = np.ma.masked_array(wl_err).astype(OPTIONS.data.dtype.real)
        return wl_val, wl_err


def get_all_wavelengths(readouts: List[ReadoutFits] | None = None) -> np.ndarray:
    """Gets all wavelengths from the readouts."""
    readouts = OPTIONS.data.readouts if readouts is None else readouts
    return np.sort(np.unique(np.concatenate(list(map(lambda x: x.wl, readouts)))))


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
    OPTIONS.fit.wls = None
    if wavelengths is None:
        return

    wavelengths = u.Quantity(wavelengths, u.um)
    if wavelengths.shape == ():
        wavelengths = wavelengths.reshape((wavelengths.size,))
    OPTIONS.fit.wls = wavelengths.flatten()
    return OPTIONS.fit.wls


def get_counts_data() -> np.ndarray:
    """Gets the number of data points for the flux,
    visibilities and closure phases."""
    counts = []
    for key in OPTIONS.fit.data:
        counts.append(getattr(OPTIONS.data, key).val.compressed().size)

    return np.array(counts)


def clear_data() -> List[str]:
    """Clears data and returns the keys of the cleared data."""
    OPTIONS.fit.wls = None
    OPTIONS.data.readouts = []

    for key in ["flux", "vis", "vis2", "t3"]:
        data = getattr(OPTIONS.data, key)
        data.val, data.err = [np.array([]) for _ in range(2)]
        if key in ["vis", "vis2"]:
            data.u, data.v = [np.array([]).reshape(1, -1) for _ in range(2)]
        elif key in "t3":
            data.u123, data.v123 = [np.array([]) for _ in range(2)]

    return ["flux", "vis", "vis2", "t3"]


def read_data(data_to_read: List[str], wavelengths: u.um) -> None:
    """Reads in the data from the keys."""
    OPTIONS.data.nB = []
    for readout in OPTIONS.data.readouts:
        for key in data_to_read:
            data = getattr(OPTIONS.data, key)
            data_readout = getattr(readout, key)
            val, err = readout.get_data_for_wavelength(
                wavelengths, key, OPTIONS.data.do_bin
            )
            if data.val.size == 0:
                data.val, data.err = val, err
            else:
                data.val = np.ma.hstack((data.val, val))
                data.err = np.ma.hstack((data.err, err))

            if key in ["vis", "vis2"]:
                if data.u.size == 0:
                    data.u = np.insert(data_readout.u, 0, 0, axis=1)
                    data.v = np.insert(data_readout.v, 0, 0, axis=1)
                else:
                    data.u = np.concatenate((data.u, data_readout.u), axis=-1)
                    data.v = np.concatenate((data.v, data_readout.v), axis=-1)

                if key == "vis2":
                    OPTIONS.data.nB.append(data_readout.u.size)

            elif key == "t3":
                uvcoords = np.stack((data_readout.u123, data_readout.v123), axis=-1)
                unique_uvcoords = np.unique(uvcoords.reshape(-1, 2), axis=0)
                ucoord, vcoord = unique_uvcoords[:, 0], unique_uvcoords[:, 1]
                index123 = np.vectorize(lambda x: np.where(ucoord == x)[0][0])(
                    data_readout.u123
                )
                if data.u123.size == 0:
                    data.u123 = np.insert(data_readout.u123, 0, 0, axis=1)
                    data.v123 = np.insert(data_readout.v123, 0, 0, axis=1)
                    data.u = np.insert(ucoord, 0, 0).reshape(1, -1)
                    data.v = np.insert(vcoord, 0, 0).reshape(1, -1)
                    data.i123 = index123 + 1
                else:
                    data.u123 = np.hstack((data.u123, data_readout.u123))
                    data.v123 = np.hstack((data.v123, data_readout.v123))
                    data.u = np.concatenate((data.u, ucoord.reshape(1, -1)), axis=-1)
                    data.v = np.concatenate((data.v, vcoord.reshape(1, -1)), axis=-1)
                    data.i123 = np.hstack((data.i123, index123 + data.i123.max() + 1))

    for key in data_to_read:
        data = getattr(OPTIONS.data, key)
        data.val = np.ma.masked_invalid(data.val)
        data.err = np.ma.masked_invalid(data.err)


# FIXME: Does not work right now -> Skews values for corr. flux
def average_data(average: bool) -> None:
    """Averages the flux data and applys a correction factor to the correlated flux."""
    if average:
        wls = OPTIONS.fit.wls
        flux, flux_err = OPTIONS.data.flux.val, OPTIONS.data.flux.err
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
                cond_lband = (limits_lband[0] < wls) & (limits_lband[1] > wls)
                cond_mband = (limits_mband[0] < wls) & (limits_mband[1] > wls)
                ind = np.where(cond_lband | cond_mband)[0]
            else:
                limits = get_band_limits(band) * u.um
                ind = np.where((limits[0] < wls) & (limits[1] > wls))[0]

            band_ratio = flux_ratio[:, index][ind]
            if band_ratio.compressed().size == band_ratio.size:
                continue

            if band_ratio.compressed().size == 0:
                flux_ratio[:, index][ind] = 1.0
            elif band_ratio.compressed().size == 1:
                flux_ratio[:, index][ind] = band_ratio.compressed()
            else:
                interp_ratios = interp1d(
                    wls[ind][~band_ratio.mask],
                    band_ratio.compressed(),
                    fill_value="extrapolate",
                )
                flux_ratio[:, index][ind] = interp_ratios(wls[ind])

        for key in ["vis", "vis2"]:
            value = getattr(OPTIONS.data, key).val
            split_indices = np.cumsum(OPTIONS.data.nbaselines[:-1])
            for index, current_slice in enumerate(split_indices):
                prev_slice = None if index == 0 else split_indices[index - 1]
                current_slice = (
                    current_slice if index != len(split_indices) - 1 else None
                )
                value[:, prev_slice:current_slice] = (
                    value[:, prev_slice:current_slice]
                    * flux_ratio[:, index][:, np.newaxis]
                )

        OPTIONS.data.flux.val = flux_averaged[:, np.newaxis]
        OPTIONS.data.flux.err = flux_err_averaged[:, np.newaxis]


def set_data(
    fits_files: List[Path] | None = None,
    wavelengths: str | u.Quantity[u.um] | None = None,
    fit_data: List[str] = ["flux", "vis", "t3"],
    weights: Dict[str, float] | str | None = None,
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
    hduls = [fits.open(fits_file) for fits_file in fits_files]
    OPTIONS.data.hduls = [copy.deepcopy(hdul) for hdul in hduls]
    [hdul.close() for hdul in hduls]

    readouts = OPTIONS.data.readouts = list(map(ReadoutFits, fits_files))
    if filter_by_array is not None:
        OPTIONS.data.readouts = list(
            filter(lambda x: x.array == filter_by_array, readouts)
        )

    OPTIONS.data.bands = list(map(lambda x: x.band, OPTIONS.data.readouts))
    if wavelengths == "all":
        wavelengths = get_all_wavelengths(OPTIONS.data.readouts)
        OPTIONS.data.do_bin = True

    if wavelengths is None:
        raise ValueError("No wavelengths given and/or not 'all' specified!")

    wavelengths = set_fit_wavelengths(wavelengths)
    read_data(data_to_read, wavelengths)
    average_data(average)

    if weights is not None:
        if weights == "ndata":
            ndata = get_counts_data()
            weights = dict(zip(OPTIONS.fit.data, (ndata / ndata.max()) ** - 1))
        OPTIONS.fit.weights = SimpleNamespace(**weights)

    return OPTIONS.data
