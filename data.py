import numpy as np
import astropy.units as u

from astropy.io import fits
from pathlib import Path
from astropy.units import Quantity
from typing import Tuple, List, Optional, Union
from scipy.interpolate import CubicSpline


# TODO: Make get_band_information method to check the band
class ReadoutFits:
    """All functionality to work with (.fits)-files"""

    def __init__(self, fits_files: List[Path]) -> None:
        """The class's constructor."""
        self.fits_files = [Path(fits_files)]\
            if not isinstance(fits_files, list) else map(Path, fits_files)
        self.wavelength = []
        self.ucoord, self.vcoord = [], []
        self.flux, self.flux_err = None, None
        self.vis, self.vis_err = None, None
        self.t3phi, self.t3phi_err = None, None

    def read_files(self):
        """Reads the data of the (.fits)-files into vectors."""
        for file in self.fits_files:
            with fits.open(Path(file)) as hdul:
                self.wavelength.extend(hdul["oi_wavelength"].data["eff_wave"].tolist())
                self.ucoord.extend(hdul["oi_vis"].data["ucoord"].tolist())
                self.vcoord.extend(hdul["oi_vis"].data["vcoord"].tolist())

                if self.flux is None:
                    self.flux = hdul["oi_flux"].data["fluxdata"]
                    self.flux_err = hdul["oi_flux"].data["fluxerr"]
                else:
                    self.flux = np.concatenate(
                        (self.flux, hdul["oi_flux"].data["fluxdata"]), axis=1)
                    self.flux_err = np.concatenate(
                        (self.flux_err, hdul["oi_flux"].data["fluxerr"]), axis=1)

                if self.vis is None:
                    self.vis = hdul["oi_vis"].data["visamp"]
                    self.vis_err = hdul["oi_vis"].data["visamperr"]
                else:
                    self.vis = np.concatenate(
                        (self.vis, hdul["oi_vis"].data["visamp"]), axis=1)
                    self.vis_err = np.concatenate(
                        (self.vis_err, hdul["oi_vis"].data["visamperr"]), axis=1)

                if self.t3phi is None:
                    self.t3phi = hdul["oi_t3"].data["t3phi"]
                    self.t3phi_err = hdul["oi_t3"].data["t3phierr"]
                else:
                    self.t3phi = np.concatenate(
                        (self.t3phi, hdul["oi_t3"].data["t3phi"]), axis=1)
                    self.t3phi_err = np.concatenate(
                        (self.t3phi_err, hdul["oi_t3"].data["t3phierr"]), axis=1)

        self.wavelength = np.array(self.wavelength)
        self.ucoord = np.array(self.ucoord)
        self.vcoord = np.array(self.vcoord)
        return self

    def get_data_for_wavelength(self, wavelength: np.ndarray):
        ...


if __name__ == "__main__":
    path = Path("/Users/scheuck/Data/reduced_data/hd142666/matisse/")
    files = ["hd_142666_2019-05-14T05_28_03:2019-05-14T06_12_59_AQUARIUS_FINAL_TARGET_INT.fits",
             "hd_142666_2022-04-21T07_18_22:2022-04-21T06_47_05_HAWAII-2RG_FINAL_TARGET_INT.fits"]
    files = [path / file for file in files]
    readout = ReadoutFits(files).read_files()
    breakpoint()
