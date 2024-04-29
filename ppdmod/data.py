from functools import partial
from typing import Optional, List, Union
from types import SimpleNamespace
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.io import fits

from .utils import get_indices
from .options import OPTIONS


class ReadoutFits:
    """All functionality to work with (.fits) or flux files.

    Parameters
    ----------
    fits_file : pathlib.Path
        The path to the (.fits) or flux file.
    """

    def __init__(self, fits_file: Path,
                 wavelength_range: Optional[u.Quantity[u.um]] = None) -> None:
        """The class's constructor."""
        self.fits_file = Path(fits_file)
        self.wavelength_range = wavelength_range
        self.read_file()

    def read_file(self) -> None:
        """Reads the data of the (.fits)-files into vectors."""
        try:
            hdul = fits.open(self.fits_file)
        # TODO: This will also have a problem opening anything else than a (.fits)-file
        except OSError:
            hdul = None
            wl, flux, flux_err = np.loadtxt(self.fits_file, unpack=True)
            self.wavelength = wl*u.um
            self.flux = SimpleNamespace(value=flux, err=flux_err)

        if hdul is not None:
            instrument = None
            if "instrume" in hdul[0].header:
                instrument = hdul[0].header["instrume"].lower()
            sci_index = OPTIONS.data.gravity.index\
                if instrument == "gravity" else None
            wl_index = 1 if instrument == "gravity" else None
            self.wavelength = (hdul["oi_wavelength", sci_index]
                               .data["eff_wave"]*u.m).to(u.um)[wl_index:]
            
            indices = slice(None)
            if self.wavelength_range is not None:
                indices = (self.wavelength_range[0] < self.wavelength) \
                    & (self.wavelength_range[1] > self.wavelength)
                self.wavelength = self.wavelength[indices]
                
            self.flux = self.read_into_namespace(
                    hdul, "flux", sci_index, wl_index, indices)
            self.t3 = self.read_into_namespace(
                    hdul, "t3", sci_index, wl_index, indices)
            self.vis = self.read_into_namespace(
                    hdul, "vis", sci_index, wl_index, indices)
            self.vis2 = self.read_into_namespace(
                    hdul, "vis2", sci_index, wl_index, indices)
            hdul.close()

    def read_into_namespace(self, hdul: fits.HDUList, key: str,
                            sci_index: Optional[int] = None,
                            wl_index: Optional[int] = None,
                            indices: Optional[int] = None) -> SimpleNamespace:
        """Reads a (.fits) Card into a SimpleNamespace."""
        try:
            data = hdul[f"oi_{key}", sci_index]
        except KeyError:
            return SimpleNamespace(value=np.array([]), err=np.array([]),
                                   ucoord=np.array([]).reshape(1, -1),
                                   vcoord=np.array([]).reshape(1, -1))

        if key == "flux":
            try:
                return SimpleNamespace(value=data.data["fluxdata"][:, indices],
                                       err=data.data["fluxerr"][:, indices])
            except KeyError:
                return SimpleNamespace(value=np.array([]), err=np.array([]))
        elif key in ["vis", "vis2"]:
            if key == "vis":
                value_key, err_key = "visamp", "visamperr"
            else:
                value_key, err_key = "vis2data", "vis2err"
            return SimpleNamespace(value=data.data[value_key][:, wl_index:][:, indices],
                                   err=data.data[err_key][:, wl_index:][:, indices],
                                   ucoord=data.data["ucoord"].reshape(1, -1),
                                   vcoord=data.data["vcoord"].reshape(1, -1))
        elif key == "t3":
            value = data.data["t3phi"][:, wl_index:][:, indices]
            err = data.data["t3phierr"][:, wl_index:][:, indices]
            u1coord, u2coord = map(lambda x: data.data[f"u{x}coord"], ["1", "2"])
            v1coord, v2coord = map(lambda x: data.data[f"v{x}coord"], ["1", "2"])

            # TODO: Check this!
            # NOTE: This should be positive as complex conjugation is applied
            # later? Also positive for Jozsef and Anthony?
            u3coord, v3coord = u1coord+u2coord, v1coord+v2coord
            u123coord = np.array([u1coord, u2coord, -u3coord])
            v123coord = np.array([v1coord, v2coord, -v3coord])
            return SimpleNamespace(value=value, err=err,
                                   u123coord=u123coord, v123coord=v123coord)

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
        # TODO: Check that the binning works proplerly for all input files
        # test it with the pionier data (for three channels)
        indices = get_indices(wavelength, array=self.wavelength,
                              window=OPTIONS.data.binning)

        data = getattr(getattr(self, key), subkey)
        if all(index.size == 0 for index in indices):
            if key == "flux":
                return np.full((wavelength.size, 1), np.nan)
            return np.full((wavelength.size, data.shape[0]), np.nan)

        if key == "flux":
            if OPTIONS.data.binning is None:
                wl_data = [[data.squeeze()[index] if index.size != 0
                           else np.full((data.shape[0],), np.nan)] for index in indices]
            else:
                wl_data = [[data.squeeze()[index].mean()] for index in indices]
        else:
            if OPTIONS.data.binning is None:
                wl_data = [data[:, index] if index.size != 0
                           else np.full((data.shape[0],), np.nan) for index in indices]
            else:
                wl_data = [data[:, index].mean(-1) for index in indices]
        return np.array(wl_data).astype(OPTIONS.data.dtype.real)


def get_all_wavelengths(readouts: Optional[List[ReadoutFits]] = None) -> np.ndarray:
    """Gets all wavelengths from the readouts."""
    readouts = OPTIONS.data.readouts if readouts is None else readouts
    wavelengths = list(map(lambda x: x.wavelength, readouts))
    return np.sort(np.unique(np.concatenate(wavelengths)))


def set_fit_wavelengths(wavelengths: Optional[u.Quantity[u.um]] = None) -> Union[str, np.ndarray]:
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


def set_fit_weights(weights: Optional[List[float]] = None, 
                    fit_data: Optional[List[str]] = None) -> None:
    """Sets the weights of the fit parameters
    from the observed data"""
    fit_data = OPTIONS.fit.data if fit_data is None else fit_data
    if weights is not None:
        wflux, wvis, wt3 = weights
    else:
        vis = OPTIONS.data.vis if OPTIONS.data.vis2.value.size == 0\
            else OPTIONS.data.vis2
        breakpoint()
        
        if vis.value.size != 0:
            nvis = vis.value.shape[1]
        else:
            nvis = 1

        if "flux" in fit_data:
            nflux = OPTIONS.data.flux.value.shape[1]
            wflux = nvis/nflux
        else:
            wflux = 1

        if "t3" in fit_data:
            nt3 = OPTIONS.data.t3.value.shape[1]
            wt3 = nvis/nt3
        else:
            wt3 = 1

        wvis = 1

    OPTIONS.fit.weights.flux = wflux
    OPTIONS.fit.weights.vis = wvis
    OPTIONS.fit.weights.nt3 = wt3


def set_data(fits_files: Optional[List[Path]] = None,
             wavelengths: Optional[Union[str, u.Quantity[u.um]]] = None,
             fit_data: Optional[List[str]] = None,
             weights: Optional[List[float]] = None,
             wavelength_range: Optional[u.Quantity[u.um]] = None,
             **kwargs) -> SimpleNamespace:
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

    OPTIONS.data.readouts = list(map(partial(ReadoutFits, wavelength_range=wavelength_range), fits_files))
    if wavelengths == "all":
        wavelengths = get_all_wavelengths(OPTIONS.data.readouts)
        OPTIONS.data.binning = None
    elif wavelengths is None:
        raise ValueError("No wavelengths given and/or not 'all' specified!")

    wavelengths = set_fit_wavelengths(wavelengths)
    for readout in OPTIONS.data.readouts:
        for key in fit_data:
            data = getattr(OPTIONS.data, key)
            data_readout = getattr(readout, key)
            value = readout.get_data_for_wavelength(wavelengths, key, "value")
            err = readout.get_data_for_wavelength(wavelengths, key, "err")

            if data.value.size == 0:
                data.value, data.err = value, err
            else:
                data.value = np.hstack((data.value, value))
                data.err = np.hstack((data.err, err))

            if key in ["vis", "vis2"]:
                data.ucoord = np.concatenate(
                        (data.ucoord, data_readout.ucoord), axis=-1)
                data.vcoord = np.concatenate(
                        (data.vcoord, data_readout.vcoord), axis=-1)

            elif key == "t3":
                if data.u123coord.size == 0:
                    data.u123coord = data_readout.u123coord
                    data.v123coord = data_readout.v123coord
                else:
                    data.u123coord = np.hstack(
                            (data.u123coord, data_readout.u123coord))
                    data.v123coord = np.hstack(
                            (data.v123coord, data_readout.v123coord))
    set_fit_weights(weights, fit_data)
    return OPTIONS.data
