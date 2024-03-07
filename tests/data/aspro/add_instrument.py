from typing import List
from pathlib import Path

import numpy as np
from astropy.io import fits


def add_instrument(fits_file: List[Path]) -> None:
    """Add the instrument name to the header of the fits file
    as as well as noise to the data.
    """
    with fits.open(fits_file, "update") as hdul:
        hdul[0].header["INSTRUME"] = "MATISSE"
        for key in ["vis", "t3"]:
            if key == "vis":
                subhdr, errhdr = "visamp", "visamperr"
            else:
                subhdr, errhdr = "t3phi", "t3phierr"

            data = hdul[f"oi_{key}"].data[subhdr]
            if key == "vis":
                noise = np.random.uniform(low=0, high=0.1, size=data.shape)
            else:
                if np.any(data > 0):
                    noise = np.random.uniform(low=0, high=5, size=data.shape)
                else:
                    noise = np.zeros(data.shape)

            hdul[f"oi_{key}"].data[errhdr] = noise
        hdul.flush()


if __name__ == "__main__":
    fits_files = list(Path.cwd().glob("*.fits"))
    for fits_file in fits_files:
        add_instrument(fits_file)
