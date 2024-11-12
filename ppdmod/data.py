from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import List

import astropy.units as u
import numpy as np
from astropy.io import fits

from .options import OPTIONS
from .utils import get_band, get_indices


class ReadoutFits:
    """All functionality to work with (.fits) or flux files.

    Parameters
    ----------
    fits_file : pathlib.Path
        The path to the (.fits) or flux file.
    """

    def __init__(
        self, fits_file: Path, wavelength_range: u.Quantity[u.um] | None = None
    ) -> None:
        """The class's constructor."""
        self.fits_file = Path(fits_file)
        self.wavelength_range = wavelength_range
        self.band = "unknown"
        self.read_file()

    def read_file(self) -> None:
        """Reads the data of the (.fits)-files into vectors."""
        try:
            hdul = fits.open(self.fits_file)
        # TODO: This will also have a problem opening anything else than a (.fits)-file
        except OSError:
            hdul = None
            wl, flux, flux_err = np.loadtxt(self.fits_file, unpack=True)
            self.wavelength = wl * u.um
            self.flux = SimpleNamespace(value=flux, err=flux_err)

        if hdul is not None:
            instrument = None
            if "instrume" in hdul[0].header:
                instrument = hdul[0].header["instrume"].lower()
            sci_index = OPTIONS.data.gravity.index if instrument == "gravity" else None
            wl_index = 1 if instrument == "gravity" else None
            self.wavelength = (
                hdul["oi_wavelength", sci_index].data["eff_wave"] * u.m
            ).to(u.um)[wl_index:]
            self.band = get_band(self.wavelength)

            indices = slice(None)
            if self.wavelength_range is not None:
                indices = (self.wavelength_range[0] < self.wavelength) & (
                    self.wavelength_range[1] > self.wavelength
                )
                self.wavelength = self.wavelength[indices]

            self.flux = self.read_into_namespace(
                hdul, "flux", sci_index, wl_index, indices
            )
            self.t3 = self.read_into_namespace(hdul, "t3", sci_index, wl_index, indices)
            self.vis = self.read_into_namespace(
                hdul, "vis", sci_index, wl_index, indices
            )
            self.vis2 = self.read_into_namespace(
                hdul, "vis2", sci_index, wl_index, indices
            )
            hdul.close()

    def read_into_namespace(
        self,
        hdul: fits.HDUList,
        key: str,
        sci_index: int | None = None,
        wl_index: int | None = None,
        indices: int | None = None,
    ) -> SimpleNamespace:
        """Reads a (.fits) Card into a SimpleNamespace."""
        try:
            data = hdul[f"oi_{key}", sci_index]
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
                    value=data.data["fluxdata"][:, indices][:, wl_index:],
                    err=data.data["fluxerr"][:, indices][:, wl_index:],
                )
            except KeyError:
                return SimpleNamespace(value=np.array([]), err=np.array([]))

        if key in ["vis", "vis2"]:
            if key == "vis":
                value_key, err_key = "visamp", "visamperr"
            else:
                value_key, err_key = "vis2data", "vis2err"
            return SimpleNamespace(
                value=data.data[value_key][:, wl_index:][:, indices],
                err=data.data[err_key][:, wl_index:][:, indices],
                ucoord=data.data["ucoord"].reshape(1, -1),
                vcoord=data.data["vcoord"].reshape(1, -1),
            )

        value = data.data["t3phi"][:, wl_index:][:, indices]
        err = data.data["t3phierr"][:, wl_index:][:, indices]
        u1coord, u2coord = map(lambda x: data.data[f"u{x}coord"], ["1", "2"])
        v1coord, v2coord = map(lambda x: data.data[f"v{x}coord"], ["1", "2"])

        u3coord, v3coord = u1coord + u2coord, v1coord + v2coord
        u123coord = np.array([u1coord, u2coord, -u3coord])
        v123coord = np.array([v1coord, v2coord, -v3coord])
        return SimpleNamespace(
            value=value, err=err, u123coord=u123coord, v123coord=v123coord
        )

    def get_data_for_wavelength(
        self,
        wavelength: u.Quantity,
        key: str,
        subkey: str,
        no_binning: bool = False,
    ) -> np.ndarray:
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
        subkey : str
            The subkey of the data to be returned.
        no_binning : bool, optional
            If the data should be binned or not.

        Returns
        -------
        numpy.ndarray
            The data for the given wavelengths.
        """
        if not no_binning:
            window = getattr(OPTIONS.data.binning, self.band)
        else:
            window = None

        indices = get_indices(wavelength, array=self.wavelength, window=window)

        data = getattr(getattr(self, key), subkey)
        if all(index.size == 0 for index in indices):
            if key == "flux":
                return np.full((wavelength.size, 1), np.nan)
            return np.full((wavelength.size, data.shape[0]), np.nan)

        if key == "flux":
            if no_binning:
                wl_data = [
                    [
                        data.flatten()[index[0]]
                        if index.size != 0
                        else np.full((data.shape[0],), np.nan).tolist()[0]
                    ]
                    for index in indices
                ]
            else:
                wl_data = [[data.flatten()[index].mean()] for index in indices]
        else:
            if no_binning:
                wl_data = [
                    data[:, index].flatten()
                    if index.size != 0
                    else np.full((data.shape[0],), np.nan)
                    for index in indices
                ]
            else:
                wl_data = [data[:, index].mean(-1) for index in indices]

        return np.array(wl_data).astype(OPTIONS.data.dtype.real)


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


def get_counts_data() -> np.ndarray[int]:
    """Gets the number of data points for the flux,
    visibilities and closure phases."""
    counts = []
    for key in OPTIONS.fit.data:
        data = getattr(OPTIONS.data, key)
        counts.append(data.value[~np.isnan(data.value)].size)

    return np.array(counts)


def set_weights(weights: List[float] | None = None) -> None:
    """Sets the weights of the fit parameters
    from the observed data"""
    if weights is None:
        weights = [1, 1, 1]

    for key, weight in zip(OPTIONS.fit.data, weights):
        setattr(OPTIONS.fit.weights, key, weight)


def set_data(
    fits_files: List[Path] | None = None,
    wavelengths: str | u.Quantity[u.um] | None = None,
    fit_data: List[str] = ["flux", "vis", "t3"],
    weights: List[float] | None = None,
    wavelength_range: u.Quantity[u.um] | None = None,
    set_std_err: List[str] = ["mband"],
    min_err: float = 0.05,
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
        The weights of the fit parameters from the observed data.
    wavelength_range : astropy.units.um, optional
        A range of wavelengths to be kept. Other wavelengths will be omitted.
    set_std_err : list of str, optional
        The data to be set the standard error from the variance of the datasets from.
        Default is ["mband"].
    min_err : float, optional
        The minimum error of the data to be kept. Will set the error to be that at least.
    """
    OPTIONS.data.readouts = []

    fit_data = OPTIONS.fit.data if fit_data is None else fit_data
    OPTIONS.fit.data = fit_data

    for key in ["flux", "vis", "vis2", "t3"]:
        data = getattr(OPTIONS.data, key)
        data.value, data.err = [np.array([]) for _ in range(2)]
        if key in ["vis", "vis2"]:
            data.ucoord, data.vcoord = [np.array([]).reshape(1, -1) for _ in range(2)]
        elif key in "t3":
            data.u123coord, data.v123coord = [np.array([]) for _ in range(2)]

    OPTIONS.fit.wavelengths = None
    if fits_files is None:
        return

    OPTIONS.data.readouts = list(
        map(partial(ReadoutFits, wavelength_range=wavelength_range), fits_files)
    )
    OPTIONS.data.bands = list(map(lambda x: x.band, OPTIONS.data.readouts))

    no_binning = False
    if wavelengths == "all":
        wavelengths = get_all_wavelengths(OPTIONS.data.readouts)
        no_binning = True
    elif wavelengths is None:
        raise ValueError("No wavelengths given and/or not 'all' specified!")

    wavelengths = set_fit_wavelengths(wavelengths)
    for readout in OPTIONS.data.readouts:
        for key in fit_data:
            data = getattr(OPTIONS.data, key)
            data_readout = getattr(readout, key)

            value = readout.get_data_for_wavelength(
                wavelengths, key, "value", no_binning
            )
            err = readout.get_data_for_wavelength(wavelengths, key, "err", no_binning)

            if key in ["vis", "vis2", "t3"]:
                ind = np.where(np.abs(err / value) < min_err)
                err[ind] = np.abs(value[ind]) * min_err

            if data.value.size == 0:
                data.value, data.err = value, err
            else:
                data.value = np.hstack((data.value, value))
                data.err = np.hstack((data.err, err))

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

            elif key == "t3":
                if data.u123coord.size == 0:
                    data.u123coord = np.insert(data_readout.u123coord, 0, 0, axis=1)
                    data.v123coord = np.insert(data_readout.v123coord, 0, 0, axis=1)
                else:
                    data.u123coord = np.hstack((data.u123coord, data_readout.u123coord))
                    data.v123coord = np.hstack((data.v123coord, data_readout.v123coord))

    for key in fit_data:
        data = getattr(OPTIONS.data, key)
        bands = np.array(list(map(get_band, wavelengths)))
        for band in set_std_err:
            ind = np.where(band == bands)[0]
            band_data = data.value[ind, :]
            band_std = np.tile(np.std(band_data, axis=0), (band_data.shape[0], 1))
            err_ind = np.where(np.abs(band_std / band_data) < min_err)
            band_std[err_ind] = np.abs(band_data[err_ind]) * min_err
            data.err[ind, :] = band_std

    set_weights(weights)
    return OPTIONS.data
