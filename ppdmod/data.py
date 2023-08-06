from typing import Union, Optional, Self, Dict, List
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.io import fits

from .utils import set_tuple_from_args, get_closest_indices
from .options import OPTIONS


class ReadoutFits:
    """All functionality to work with (.fits)-files"""

    def __init__(self, fits_file: Path) -> None:
        """The class's constructor."""
        self.fits_file = Path(fits_file)
        self.read_file()

    def read_file(self) -> Self:
        """Reads the data of the (.fits)-files into vectors."""
        with fits.open(Path(self.fits_file)) as hdul:
            self.wavelength = (hdul["oi_wavelength"].data["eff_wave"]*u.m).to(u.um)
            self.ucoord = hdul["oi_vis"].data["ucoord"]
            self.vcoord = hdul["oi_vis"].data["vcoord"]
            self.flux = hdul["oi_flux"].data["fluxdata"]
            self.flux_err = hdul["oi_flux"].data["fluxerr"]
            self.vis = hdul["oi_vis"].data["visamp"]
            self.vis_err = hdul["oi_vis"].data["visamperr"]
            self.t3phi = hdul["oi_t3"].data["t3phi"]
            self.t3phi_err = hdul["oi_t3"].data["t3phierr"]
            self.u1coord = hdul["oi_t3"].data["u1coord"]
            self.u2coord = hdul["oi_t3"].data["u2coord"]
            self.u3coord = -(self.u1coord+self.u2coord)
            self.v1coord = hdul["oi_t3"].data["v1coord"]
            self.v2coord = hdul["oi_t3"].data["v2coord"]
            self.v3coord = -(self.v1coord+self.v2coord)
            self.u123coord = [self.u1coord, self.u2coord, self.u3coord]
            self.v123coord = [self.v1coord, self.v2coord, self.v3coord]
        return self

    def get_data_for_wavelengths(self,
                                *wavelengths, key: str) -> Dict[str, np.ndarray]:
        """Gets the data for the given wavelengths."""
        indices = get_closest_indices(*wavelengths, array=self.wavelength)
        keys = [str(wavelength.value) if isinstance(wavelength, u.Quantity)
                else str(wavelength) for wavelength in set_tuple_from_args(*wavelengths)]
        data = {keys[count]: getattr(self, key)[:, index].squeeze().T
                for count, index in enumerate(indices)}
        return {key: value for key, value in data.items() if value.size != 0}


def set_fit_wavelengths(*wavelengths: u.um) -> None:
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
    OPTIONS["fit.wavelengths"] = wavelengths.to(u.um).flatten()


def get_data(*fits_files: Optional[Union[List[Path], Path]]) -> None:
    """Sets the data as a global variable from the input files.

    If called without parameters or recalled, it will clear the data.
    """
    OPTIONS["data.readouts"] = []
    OPTIONS["data.total_flux"],\
        OPTIONS["data.total_flux_error"] = [], []
    OPTIONS["data.correlated_flux"],\
        OPTIONS["data.correlated_flux_error"] = [], []
    OPTIONS["data.closure_phase"],\
        OPTIONS["data.closure_phase_error"] = [], []

    if not fits_files:
        return

    fits_files = set_tuple_from_args(*fits_files)
    readouts = OPTIONS["data.readouts"] =\
        [ReadoutFits(file) for file in fits_files]
    if not OPTIONS["fit.wavelengths"]:
        raise ValueError("Fitting wavelengths must be specified!")

    wavelengths = OPTIONS["fit.wavelengths"]
    for readout in readouts:
        OPTIONS["data.total_flux"].append(
            readout.get_data_for_wavelengths(wavelengths, key="flux"))
        OPTIONS["data.total_flux_error"].append(
            readout.get_data_for_wavelengths(wavelengths, key="flux_err"))
        OPTIONS["data.correlated_flux"].append(
            readout.get_data_for_wavelengths(wavelengths, key="vis"))
        OPTIONS["data.correlated_flux_error"].append(
            readout.get_data_for_wavelengths(wavelengths, key="vis_err"))
        OPTIONS["data.closure_phase"].append(
            readout.get_data_for_wavelengths(wavelengths, key="t3phi"))
        OPTIONS["data.closure_phase_error"].append(
            readout.get_data_for_wavelengths(wavelengths, key="t3phi_err"))


if __name__ == "__main__":
    path = Path("/Users/scheuck/Data/reduced_data/hd142666/matisse/")
    file = "hd_142666_2019-05-14T05_28_03:2019-05-14T06_12_59_AQUARIUS_FINAL_TARGET_INT.fits"
    readout = ReadoutFits(path / file).read_file()
    data = readout.get_data_for_wavelengths(readout.wavelength[50:52], "t3phi")
    breakpoint()
