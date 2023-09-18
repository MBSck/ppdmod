from pathlib import Path

from astropy.io import fits

for fits_file in list(Path().glob("*AQUARIUS*")):
    with fits.open(fits_file, "update") as hdul:
        t3phi = hdul["oi_t3"].data["t3phi"]
        hdul["oi_t3"].data["t3phi"] = -t3phi
