import numpy as np

from astropy.io import fits
from pathlib import Path
from typing import List


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
            self.u1coord = hdul["oi_t3"].data["u1coord"]
            self.u2coord = hdul["oi_t3"].data["u2coord"]
            self.v1coord = hdul["oi_t3"].data["v1coord"]
            self.v2coord = hdul["oi_t3"].data["v2coord"]
        return self

    def get_data_for_wavelength(self, wavelengths: np.ndarray, key: str):
        """Gets the data for the given wavelengths."""
        indicies = [np.where(self.wavelength == wavelength)[0]
                    for wavelength in wavelengths]
        data = {str(wavelengths[count]): getattr(self, key)[:, index].squeeze().T
                for count, index in enumerate(indicies)}
        return {key: value for key, value in data.items() if value.size != 0}


if __name__ == "__main__":
    path = Path("/Users/scheuck/Data/reduced_data/hd142666/matisse/")
    file = "hd_142666_2019-05-14T05_28_03:2019-05-14T06_12_59_AQUARIUS_FINAL_TARGET_INT.fits"
    readout = ReadoutFits(path / file).read_files()
    data = readout.get_data_for_wavelength(readout.wavelength[50:52], "t3phi")
    breakpoint()
