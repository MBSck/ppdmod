from pathlib import Path

import numpy as np
from ppdmod.basic_components import Ring, StarHaloRing
from ppdmod.data import set_data, get_all_wavelengths
from ppdmod.parameter import Parameter
from ppdmod.options import STANDARD_PARAMETERS, OPTIONS
from ppdmod.plot import plot_components
from ppdmod.utils import compute_photometric_slope


if __name__ == "__main__":
    DATA_DIR = Path("../data/gravity")
    OPTIONS.model.output = "non-physical"
    fits_files = list((DATA_DIR).glob("*fits"))
    data = set_data(fits_files, wavelengths="all", fit_data=["vis2", "t3"])
    
    wavelength = get_all_wavelengths()
    ks = Parameter(**STANDARD_PARAMETERS.exp)
    ks.value = compute_photometric_slope(wavelength, 6500)
    ks.wavelength = wavelength
    ks.free = False
    
    result_dir = Path("results/pionier")
    ring = Ring(rin=2, asymmetric=True, a=1, phi=30)

    # FSCMa
    # shlr = StarHaloRing(
    #     fs=0.42, fc=0.55, flor=1.0,
    #     la=0.98, lkr=-0.26,
    #     ks=ks, kc=-4.12,
    #     inc=0.63, pa=1.2/np.pi*180,
    #     a=0.996393496566492,
    #     phi=100.40771131249006)

    plot_components(ring, 512, 0.02, 3.5, savefig=result_dir / "test.png", save_as_fits=False)
    plot_components(ring, 512, 0.02, 3.5, savefig=result_dir / "test.fits", save_as_fits=True)
