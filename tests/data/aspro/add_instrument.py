from typing import List
from pathlib import Path

from astropy.io import fits


def add_instrument(fits_file: List[Path]) -> None:
    with fits.open(fits_file, "update") as hdul:
        hdul[0].header["INSTRUME"] = "MATISSE"
        hdul.flush()


if __name__ == "__main__":
    fits_files = list(Path.cwd().glob("*.fits"))
    for fits_file in fits_files:
        add_instrument(fits_file)
