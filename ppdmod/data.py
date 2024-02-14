from typing import Optional, List
from types import SimpleNamespace
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.io import fits

from .utils import get_closest_indices
from .options import OPTIONS


class ReadoutFits:
    """All functionality to work with (.fits)-files"""

    def __init__(self, fits_file: Path) -> None:
        """The class's constructor."""
        self.fits_file = Path(fits_file)
        self.read_file()

    def read_file(self) -> None:
        """Reads the data of the (.fits)-files into vectors."""
        with fits.open(Path(self.fits_file)) as hdul:
            instrument = hdul[0].header["instrume"].lower()
            sci_index = OPTIONS.data.gravity.index\
                if instrument == "gravity" else None
            wl_index = 1 if instrument == "gravity" else None
            self.wavelength = (hdul["oi_wavelength", sci_index]
                               .data["eff_wave"]*u.m).to(u.um)[wl_index:]
            self.flux = self.read_into_namespace(
                    hdul, "flux", sci_index, wl_index)
            self.t3 = self.read_into_namespace(
                    hdul, "t3", sci_index, wl_index)
            self.vis = self.read_into_namespace(
                    hdul, "vis", sci_index, wl_index)
            self.vis2 = self.read_into_namespace(
                    hdul, "vis2", sci_index, wl_index)

    def read_into_namespace(self, hdul: fits.HDUList, key: str,
                            sci_index: Optional[int],
                            wl_index: Optional[int]) -> SimpleNamespace:
        """Reads a (.fits) Card into a SimpleNamespace."""
        try:
            data = hdul[f"oi_{key}", sci_index]
        except KeyError:
            return SimpleNamespace(value=None, err=None,
                                   ucoord=None, vcoord=None)

        if key == "flux":
            return SimpleNamespace(value=data.data["fluxdata"],
                                   err=data.data["fluxerr"])
        elif key in ["vis", "vis2"]:
            if key == "vis":
                value_key, err_key = "visamp", "visamperr"
            else:
                value_key, err_key = "vis2data", "vis2err"
            return SimpleNamespace(value=data.data[value_key][:, wl_index:],
                                   err=data.data[err_key][:, wl_index:],
                                   ucoord=data.data["ucoord"],
                                   vcoord=data.data["vcoord"])
        elif key == "t3":
            value = data.data["t3phi"][:, wl_index:]
            err = data.data["t3phierr"][:, wl_index:]
            u1coord, u2coord = map(lambda x: data.data[f"u{x}coord"], ["1", "2"])
            v1coord, v2coord = map(lambda x: data.data[f"v{x}coord"], ["1", "2"])

            # TODO: Check this!
            # NOTE: This should be positive as complex conjugation is applied
            # later?
            u3coord, v3coord = -(u1coord+u2coord), -(v1coord+v2coord)
            u123coord = np.array([u1coord, u2coord, u3coord])
            v123coord = np.array([v1coord, v2coord, v3coord])
            return SimpleNamespace(value=value, err=err,
                                   u123coord=u123coord, v123coord=v123coord)

    # TODO: Make sure to fill this up to the correct shape if vis is too small
    # or t3.
    def get_data_for_wavelength(
            self, wavelength: u.Quantity,
            key: str, subkey: str) -> np.ndarray:
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

        Returns
        -------
        numpy.ndarray
            The data for the given wavelengths.
        """
        indices = get_closest_indices(
                wavelength, array=self.wavelength,
                window=OPTIONS.data.binning.window)

        data = getattr(getattr(self, key), subkey)
        if all(index.size == 0 for index in indices):
            if key == "flux":
                return np.full((wavelength.size, 1), np.nan)
            return np.full((wavelength.size, data.shape[0]), np.nan)

        if key == "flux":
            data = [[data.squeeze()[index].mean()] for index in indices]
        else:
            data = [data[:, index].mean(1) for index in indices]
        return np.array(data).astype(OPTIONS.data.dtype.real)


def set_fit_wavelengths(
        wavelength: Optional[u.Quantity[u.um]] = None) -> None:
    """Sets the wavelengths to be fitted for as a global option.

    If called without parameters or recalled, it will clear the
    fit wavelengths.
    """
    OPTIONS.fit.wavelengths = []
    if wavelength is None:
        return

    wavelength = u.Quantity(wavelength, u.um)
    if wavelength.shape == ():
        wavelength = wavelength.reshape((wavelength.size,))
    OPTIONS.fit.wavelengths = wavelength.flatten()


def set_fit_weights(weights: Optional[List[float]] = None) -> None:
    """Sets the weights of the fit parameters
    from the observed data"""
    if weights is not None:
        wflux, wvis, wt3 = weights
    else:
        if OPTIONS.data.vis2.value.size == 0:
            nvis = OPTIONS.data.vis.value.shape[1]
        else:
            nvis = OPTIONS.data.vis2.value.shape[1]

        if "flux" in OPTIONS.fit.data:
            nflux = OPTIONS.data.flux.value.shape[1]
            wflux = nvis/nflux
        else:
            wflux = 1

        if "t3" in OPTIONS.fit.data:
            nt3 = OPTIONS.data.t3.value.shape[1]
            wt3 = nvis/nt3
        else:
            wt3 = 1

        wvis = 1

    OPTIONS.fit.weights.flux = wflux
    OPTIONS.fit.weights.vis = wvis
    OPTIONS.fit.weights.nt3 = wt3


def set_data(fits_files: Optional[List[Path]] = None,
             wavelength: Optional[u.Quantity[u.um]] = None,
             fit_data: Optional[List[str]] = None) -> None:
    """Sets the data as a global variable from the input files.

    If called without parameters or recalled, it will clear the data.

    Parameters
    ----------
    fits_files : list of Path
        The paths to the MATISSE (.fits)-files.
    wavelength : astropy.units.um
        The wavelengths to be fitted.
    fit_data : list of str, optional
    set_weights : bool, optional
        If True will set the weights of the fit parameters from the
        sizes of the input grids.

    """
    OPTIONS.data.readouts = []
    wavelength = OPTIONS.fit.wavelengths if wavelength\
        is None else wavelength
    fit_data = OPTIONS.fit.data if fit_data is None\
        else fit_data

    for key in ["flux", "vis", "vis2", "t3"]:
        data = getattr(OPTIONS.data, key)
        data.value, data.err = [np.array([]).astype(OPTIONS.data.dtype.real)
                                for _ in range(2)]
        if key in ["vis", "vis2"]:
            data.ucoord, data.vcoord = [np.array([]).astype(OPTIONS.data.dtype.real)
                                        for _ in range(2)]
        elif key in "t3":
            data.u123coord, data.v123coord =[np.array([]).astype(OPTIONS.data.dtype.real)
                                             for _ in range(2)]

    if fits_files is None:
        return

    OPTIONS.data.readouts = list(map(ReadoutFits, fits_files))
    for readout in OPTIONS.data.readouts:
        for key in fit_data:
            data = getattr(OPTIONS.data, key)
            data_readout = getattr(readout, key)
            value = readout.get_data_for_wavelength(
                    wavelength, key, "value")
            err = readout.get_data_for_wavelength(
                    wavelength, key, "err")

            if data.value.size == 0:
                data.value, data.err = value, err
            else:
                data.value = np.hstack((data.value, value))
                data.err = np.hstack((data.err, err))

            if key in ["vis", "vis2"]:
                data.ucoord = np.concatenate(
                        (data.ucoord, data_readout.ucoord))
                data.vcoord = np.concatenate(
                        (data.vcoord, data_readout.vcoord))

            elif key == "t3":
                if data.u123coord.size == 0:
                    data.u123coord = data_readout.u123coord
                    data.v123coord = data_readout.v123coord
                else:
                    data.u123coord = np.hstack(
                            (data.u123coord, data_readout.u123coord))
                    data.v123coord = np.hstack(
                            (data.v123coord, data_readout.v123coord))
