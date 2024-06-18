from pathlib import Path

import astropy.units as u
import numpy as np
from ppdmod.basic_components import AsymmetricTempGradient, AsymmetricGreyBody, Ring, Star, StarHaloRing
from ppdmod.data import set_data, get_all_wavelengths
from ppdmod.parameter import Parameter
from ppdmod.options import STANDARD_PARAMETERS, OPTIONS
from ppdmod.plot import plot_components
from ppdmod.utils import compute_photometric_slope, compute_t3, \
    compute_vis, compute_stellar_radius, load_data, qval_to_opacity, get_opacity


if __name__ == "__main__":
    DATA_DIR = Path("../data/fits/hd142527/")
    fits_files = list((DATA_DIR).glob("*HAW*fits"))
    data = set_data(fits_files, wavelengths="all", fit_data=["vis2", "t3"])
    
    wl = get_all_wavelengths()
    ks = Parameter(**STANDARD_PARAMETERS.exp)
    ks.value = compute_photometric_slope(wl, 6500)
    ks.wavelength = wl
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
    dim, pixel_size, wl = 4096, 0.02, 3.5
    plot_components(model, dim, pixel_size, wl, savefig=test_dir / f"{model.shortname}.png", save_as_fits=False, norm=1)
    plot_components(model, dim, pixel_size, wl, savefig=test_dir / f"{model.shortname}.fits", save_as_fits=True)

    DATA_DIR_NBAND = Path("../data")
    weights = np.array([73.2, 8.6, 0.6, 14.2, 2.4, 1.0])/100
    names = ["pyroxene", "forsterite", "enstatite", "silica"]
    fmaxs = [1.0, 1.0, 1.0, 0.7]
    sizes = [[1.5], [0.1], [0.1, 1.5], [0.1, 1.5]]

    _, opacity = get_opacity(
        DATA_DIR_NBAND, weights, sizes, names, "qval", wl, fmaxs)
    cont_opacity_file = DATA_DIR_NBAND / "qval" / "Q_amorph_c_rv0.1.dat"
    wl_cont, cont_opacity = load_data(cont_opacity_file, load_func=qval_to_opacity)
    cont_opacity = np.interp(wl, wl_cont, cont_opacity)

    distance, eff_temp = 157.3, 6500
    eff_radius = compute_stellar_radius(10**0.96, eff_temp).value

    wl_flux, flux = load_data(DATA_DIR_NBAND / "flux" / "hd142527" / "HD142527_stellar_model.txt")
    star_flux = Parameter(**STANDARD_PARAMETERS.f)
    star_flux.wavelength, star_flux.value = wl_flux, flux
    star = Star(f=star_flux)
    atg = AsymmetricTempGradient(rin=1.5, rout=2, dist=distance, eff_temp=eff_temp, eff_radius=eff_radius,
                                 inc=0.5, pa=33, q=0.5, temp0=1500, p=0.5, sigma0=1e-4, r0=1,
                                 kappa_abs=opacity, kappa_cont=cont_opacity, cont_weight=0.4, c1=0.5, s1=0.5)
    model = [star, atg]
    model_names = "_".join([m.shortname for m in model])

    plot_components(model, dim, pixel_size, wl, savefig=test_dir / f"{model_names}.png", save_as_fits=False, norm=0.5)
    plot_components(model, dim, pixel_size, wl, savefig=test_dir / f"{model_names}.fits", save_as_fits=True)
