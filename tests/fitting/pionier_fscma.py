from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
from ppdmod import data
from ppdmod.basic_components import Ring, StarHaloRing
from ppdmod.parameter import Parameter
from ppdmod.options import STANDARD_PARAMETERS, OPTIONS
from ppdmod.utils import compute_photometric_slope


if __name__ == "__main__":
    DATA_DIR = Path("../data/gravity")
    OPTIONS.model.output = "non-physical"
    fits_files = list((DATA_DIR).glob("*fits"))
    data.set_data(fits_files, wavelengths="all", fit_data=["vis2", "t3"])
    
    wavelength = data.get_all_wavelengths()
    ks = Parameter(**STANDARD_PARAMETERS.exp)
    ks.value = compute_photometric_slope(wavelength, 6500)
    ks.wavelength = wavelength
    ks.free = False
    
    ring = Ring(rin=2)
    image = ring.compute_image(512, 0.02, 1.68)
    plt.imshow(image[0])
    plt.show()
    plt.close()

    shlr = StarHaloRing(
        fs=0.42, fc=0.55, flor=1.0,
        la=0.98, lkr=-0.26,
        ks=ks, kc=-4.12,
        inc=0.63, pa=1.2*u.rad.to(u.deg),
        a=0.996393496566492,
        phi=100.40771131249006)
    image = shlr.compute_image(512, 0.02, 1.68)
    plt.imshow(image[0])
    plt.show()
    plt.close()
