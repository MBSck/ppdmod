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
    test_dir = Path("results/pionier/test")
    test_dir.mkdir(parents=True, exist_ok=True)

    ring = Ring(rin=2, rout=2.5, inc=0.5, pa=33, has_outer_radius=True, thin=False, asymmetric=True, c1=0.5, s1=0.5)

    # gauss = Gaussian(hlr=4.59933786, inc=0.63, pa=68.75493541569878)
    # gauss_lor = GaussLorentzian(flor=1., hlr=4.59933786*0.63,
    #                             inc=0.63, pa=68.75493541569878)
    #
    # # NOTE: FSCMa
    # shlr = StarHaloRing(
    #     fs=0.42, fc=0.55, flor=1.0,
    #     la=0.98, lkr=-0.26,
    #     ks=ks, kc=-4.12,
    #     inc=0.63, pa=1.2/np.pi*180,
    #     c1=-0.18, s1=0.98)

    model = ring

    # vis = compute_vis(shlr.compute_complex_vis(data.vis2.ucoord, data.vis2.vcoord, [3.5]*u.um))
    # t3 = compute_t3(shlr.compute_complex_vis(data.t3.u123coord, data.t3.v123coord, [3.5]*u.um))
    dim, pixel_size, wl = 4096, 0.02, 1.68
    plot_components(model, dim, pixel_size, wl, savefig=test_dir / f"{model.shortname}.png", save_as_fits=False, norm=1)
    plot_components(model, dim, pixel_size, wl, savefig=test_dir / f"{model.shortname}.fits", save_as_fits=True)
