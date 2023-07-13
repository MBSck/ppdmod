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

    def __init__(self, fits_file: List[Path]) -> None:
        """The class's constructor."""
        self.fits_file = Path(fits_file)
        self.read_files()

    def read_files(self):
        """Reads the data of the (.fits)-files into vectors."""
        with fits.open(Path(self.fits_file)) as hdul:
            self.wavelength = hdul["oi_wavelength"].data["eff_wave"]
            self.ucoord = hdul["oi_vis"].data["ucoord"]
            self.vcoord = hdul["oi_vis"].data["vcoord"]
            self.flux = hdul["oi_flux"].data["fluxdata"]
            self.flux_err = hdul["oi_flux"].data["fluxerr"]
            self.vis = hdul["oi_vis"].data["visamp"]
            self.vis_err = hdul["oi_vis"].data["visamperr"]
            self.t3phi = hdul["oi_t3"].data["t3phi"]
            self.t3phi_err = hdul["oi_t3"].data["t3phierr"]
        return self

    def get_data_for_wavelength(self, wavelengths: np.ndarray, key: str):
        """Gets the data for the given wavelengths."""
        indicies = np.where(np.isin(self.wavelength, wavelengths))
        return getattr(self, key)[:, indicies].squeeze().T


if __name__ == "__main__":
    path = Path("/Users/scheuck/Data/reduced_data/hd142666/matisse/")
    file = "hd_142666_2019-05-14T05_28_03:2019-05-14T06_12_59_AQUARIUS_FINAL_TARGET_INT.fits"
    readout = ReadoutFits(path / file).read_files()
    data = readout.get_data_for_wavelength(readout.wavelength[50:52], "t3phi")
    breakpoint()
