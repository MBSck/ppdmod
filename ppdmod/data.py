from typing import Optional, List
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

    def read_file(self):
        """Reads the data of the (.fits)-files into vectors."""
        with fits.open(Path(self.fits_file)) as hdul:
            instrument = hdul[0].header["instrume"].lower()
            sci_index = OPTIONS.data.gravity.index\
                if instrument == "gravity" else None
            wl_index = 1 if instrument == "gravity" else None
            self.wavelength = (hdul["oi_wavelength", sci_index]
                               .data["eff_wave"]*u.m).to(u.um)[wl_index:]
            self.ucoord = hdul["oi_vis2", sci_index].data["ucoord"]
            self.vcoord = hdul["oi_vis2", sci_index].data["vcoord"]

            try:
                self.flux = hdul["oi_flux", sci_index].data["fluxdata"]
                self.flux_err = hdul["oi_flux", sci_index].data["fluxerr"]
            except KeyError:
                self.flux = None
                self.flux_err = None

            try:
                self.vis = hdul["oi_vis", sci_index].data["visamp"][:, wl_index:]
                self.vis_err = hdul["oi_vis", sci_index].data["visamperr"][:, wl_index:]
            except KeyError:
                self.vis = None
                self.vis_err = None

            self.vis2 = hdul["oi_vis2", sci_index].data["vis2data"][:, wl_index:]
            self.vis2_err = hdul["oi_vis2", sci_index].data["vis2err"][:, wl_index:]
            self.t3phi = hdul["oi_t3", sci_index].data["t3phi"][:, wl_index:]
            self.t3phi_err = hdul["oi_t3", sci_index].data["t3phierr"][:, wl_index:]
            self.u1coord = hdul["oi_t3", sci_index].data["u1coord"]
            self.u2coord = hdul["oi_t3", sci_index].data["u2coord"]
            self.u3coord = -(self.u1coord+self.u2coord)
            self.v1coord = hdul["oi_t3", sci_index].data["v1coord"]
            self.v2coord = hdul["oi_t3", sci_index].data["v2coord"]
            self.v3coord = -(self.v1coord+self.v2coord)
            self.u123coord = np.array([self.u1coord, self.u2coord, self.u3coord])
            self.v123coord = np.array([self.v1coord, self.v2coord, self.v3coord])
        return self

    def get_data_for_wavelength(
            self, wavelength, key: str) -> np.ndarray:
        """Gets the data for the given wavelengths."""
        indices = list(get_closest_indices(
            wavelength, array=self.wavelength,
            window=OPTIONS.data.binning.window).values())

        if not indices:
            return np.array([])
        indices = indices[0]

        wl_data = getattr(self, key)[:, indices]
        if wl_data.shape[0] == 1 or len(wl_data.shape) == 1:
            wl_data = wl_data.mean()
        else:
            wl_data = wl_data.mean(1)
        return wl_data if isinstance(wl_data, (list, np.ndarray))\
            else np.array([wl_data])


def set_fit_wavelengths(
        wavelengths: Optional[u.Quantity[u.um]] = None) -> None:
    """Sets the wavelengths to be fitted for as a global option.

    If called without parameters or recalled, it will clear the
    fit wavelengths.
    """
    OPTIONS.fit.wavelengths = []
    if wavelengths is None:
        return

    wavelengths = u.Quantity(wavelengths, u.um)
    OPTIONS.fit.wavelengths = wavelengths.flatten()


def set_data(fits_files: Optional[List[Path]] = None,
             fit_data: Optional[List[str]] = None,
             wavelengths: Optional[u.Quantity[u.um]] = None) -> None:
    """Sets the data as a global variable from the input files.

    If called without parameters or recalled, it will clear the data.

    Parameters
    ----------
    fits_files : list of Path
        The paths to the MATISSE (.fits)-files.
    wavelengths : astropy.units.um
        The wavelengths to be fitted.
    """
    OPTIONS.data.readouts = []
    wavelengths = OPTIONS.fit.wavelengths if wavelengths\
        is None else wavelengths
    fit_data = OPTIONS.fit.data if fit_data is None\
        else fit_data

    keys = ["flux", "vis", "vis2", "t3phi"]
    for key in keys:
        data = getattr(OPTIONS.data, key)
        data.value = [[] for _ in wavelengths]
        data.err = [[] for _ in wavelengths]
        if key in ["vis", "vis2"]:
            data.ucoord = [[] for _ in wavelengths]
            data.vcoord = [[] for _ in wavelengths]
        elif key == "t3phi":
            data.u123coord = [[] for _ in wavelengths]
            data.v123coord = [[] for _ in wavelengths]

    if fits_files is None:
        return

    readouts = OPTIONS.data.readouts = list(map(ReadoutFits, fits_files))

    for readout in readouts:
        for key in fit_data:
            data = getattr(OPTIONS.data, key)
            for index, wavelength in enumerate(wavelengths):
                values = readout.get_data_for_wavelength(
                        wavelength, key=key)

                if values.size == 0:
                    continue

                values_err = readout.get_data_for_wavelength(
                        wavelength, key=f"{key}_err")

                data.value[index].extend(values)
                data.err[index].extend(values_err)

                if key in ["vis", "vis2"]:
                    data.ucoord[index].extend(readout.ucoord)
                    data.vcoord[index].extend(readout.vcoord)
                elif key == "t3phi":
                    data.u123coord[index].extend(readout.u123coord)
                    data.v123coord[index].extend(readout.v123coord)

    keys = ["flux", "vis", "vis2", "t3phi"]
    for key in keys:
        data = getattr(OPTIONS.data, key)
        if not any(np.any(value) for value in vars(data).values()):
            continue

        data.value = [np.array(value) for value in data.value]
        data.err = [np.array(value) for value in data.err]

        if key in ["vis", "vis2"]:
            data.ucoord = [np.array(value) for value in data.ucoord]
            data.vcoord = [np.array(value) for value in data.vcoord]
        elif key == "t3phi":
            data.u123coord =\
                [np.array([np.concatenate((value[i::3])) for i in range(3)])
                 for value in data.u123coord]
            data.v123coord =\
                [np.array([np.concatenate((value[i::3])) for i in range(3)])
                 for value in data.v123coord]
