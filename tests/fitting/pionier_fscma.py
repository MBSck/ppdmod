from pathlib import Path

import astropy.units as u
import numpy as np
from ppdmod.basic_components import Ring, StarHaloRing, Gaussian, GaussLorentzian
from ppdmod.data import set_data, get_all_wavelengths
from ppdmod.parameter import Parameter
from ppdmod.options import STANDARD_PARAMETERS, OPTIONS
from ppdmod.plot import plot_components
from ppdmod.utils import compute_photometric_slope, compute_t3, compute_vis


if __name__ == "__main__":
    DATA_DIR = Path("../data/fits/hd142527/")
    OPTIONS.model.output = "non-physical"
    fits_files = list((DATA_DIR).glob("*HAW*fits"))
    data = set_data(fits_files, wavelengths="all", fit_data=["vis2", "t3"])
    
    wavelength = get_all_wavelengths()
    ks = Parameter(**STANDARD_PARAMETERS.exp)
    ks.value = compute_photometric_slope(wavelength, 6500)
    ks.wavelength = wavelength
    ks.free = False
    
    result_dir = Path("results/pionier")
    ring = Ring(rin=8.36941905, inc=0.63, pa=68.75493541569878,
                asymmetric=True, a=0.996393496566492, phi=-10.407711312490056)

    gauss = Gaussian(hlr=4.59933786, inc=0.63, pa=68.75493541569878)
    gauss_lor = GaussLorentzian(flor=1., hlr=4.59933786*0.63,
                                inc=0.63, pa=68.75493541569878)

    # NOTE: FSCMa
    shlr = StarHaloRing(
        fs=0.42, fc=0.55, flor=1.0,
        la=0.98, lkr=-0.26,
        ks=ks, kc=-4.12,
        inc=0.63, pa=1.2/np.pi*180,
        a=0.996393496566492, phi=-10)

    vis = compute_vis(shlr.compute_complex_vis(data.vis2.ucoord, data.vis2.vcoord, [3.5]*u.um))
    t3 = compute_t3(shlr.compute_complex_vis(data.t3.u123coord, data.t3.v123coord, [3.5]*u.um))
    breakpoint()
    dim, pixel_size, wl = 1024, 0.02, 3.5
    plot_components(ring, dim, pixel_size, wl, savefig=result_dir / "test.png", save_as_fits=False)
    plot_components(ring, dim, pixel_size, wl, savefig=result_dir / "test.fits", save_as_fits=True)
