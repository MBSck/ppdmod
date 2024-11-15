from pathlib import Path

import numpy as np
from ppdmod.basic_components import AsymmetricGreyBody, GreyBody, Ring, Star, StarHaloRing
from ppdmod.data import set_data, get_all_wavelengths
from ppdmod.parameter import Parameter
from ppdmod.options import STANDARD_PARAMS, OPTIONS
from ppdmod.plot import plot_components
from ppdmod.utils import compute_photometric_slope, compute_t3, \
    compute_vis, compute_stellar_radius, load_data, qval_to_opacity, get_opacity


if __name__ == "__main__":
    DATA_DIR = Path("../data/fits/hd142527/")
    fits_files = list((DATA_DIR).glob("*HAW*fits"))
    data = set_data(fits_files, wavelengths="all", fit_data=["vis2", "t3"])
    wl = get_all_wavelengths()

    result_dir = Path("results/")
    test_dir = Path("results/test")
    test_dir.mkdir(parents=True, exist_ok=True)

    OPTIONS.model.modulation = 3
    dim, pixel_size, wl = 4096, 0.02, 3.5
    width = 0.5
    inc, pa = 0.5, 33
    c1, s1 = 1, 0
    c2, s2 = 0, 0
    c3, s3 = 0, 0
         # ("cm_ring_rin2_rout25_inc05_pa33_c05_s05_w0_extended.fits", 2, 3.5, 0.5, 33, 0.5, [0.5, 1], [0.5, 0]),
    ring = Ring(rin=2, has_outer_radius=False, width=width, inc=0.5, pa=33,
                thin=False, asymmetric=True, c1=c1, s1=s1, c2=c2, s2=s2, c3=c3, s3=s3)
    plot_components(ring, dim, pixel_size, wl, zoom=5,
                    savefig=test_dir / f"{ring.shortname}.png", save_as_fits=False, norm=1)
    plot_components(ring, dim, pixel_size, wl, savefig=test_dir / f"{ring.shortname}.fits", save_as_fits=True)

    # DATA_DIR_NBAND = Path("../data")
    # weights = np.array([73.2, 8.6, 0.6, 14.2, 2.4, 1.0])/100
    # names = ["pyroxene", "forsterite", "enstatite", "silica"]
    # sizes = [[1.5], [0.1], [0.1, 1.5], [0.1, 1.5]]
    #
    # wl_opacity, opacity = get_opacity(
    #     DATA_DIR_NBAND, weights, sizes, names, "boekel")
    #
    # cont_opacity_file = DATA_DIR_NBAND / "qval" / "Q_amorph_c_rv0.1.dat"
    # # cont_opacity_file = DATA_DIR / "qval" / "Q_iron_0.10um_dhs_0.7.dat",
    # wl_cont, cont_opacity = load_data(cont_opacity_file, load_func=qval_to_opacity)
    #
    # kappa_abs = Parameter(**STANDARD_PARAMETERS.kappa_abs)
    # kappa_abs.value, kappa_abs.wavelength = opacity, wl_opacity
    # kappa_cont = Parameter(**STANDARD_PARAMETERS.kappa_cont)
    # kappa_cont.value, kappa_cont.wavelength = cont_opacity, wl_cont
    #
    # inc, pa = 0.5, 33
    # c, s = 0.5, 1
    # distance, eff_temp = 158.51, 6500
    # eff_radius = compute_stellar_radius(10**1.35, eff_temp).value
    #
    # wl_flux, flux = load_data(DATA_DIR_NBAND / "flux" / "hd142527" / "HD142527_stellar_model.txt")
    # star_flux = Parameter(**STANDARD_PARAMETERS.f)
    # star_flux.wavelength, star_flux.value = wl_flux, flux
    # star = Star(f=star_flux)
    # atg = AsymmetricGreyBody(rin=1.5, rout=2, dist=distance, eff_temp=eff_temp, eff_radius=eff_radius,
    #                          inc=inc, pa=pa, p=0.5, sigma0=1e-4, r0=1, c1=c, s1=s,
    #                          kappa_abs=kappa_abs, kappa_cont=kappa_cont, cont_weight=0.9)
    # atg2 = AsymmetricGreyBody(rin=3, rout=5, dist=distance, eff_temp=eff_temp, eff_radius=eff_radius,
    #                           inc=inc, pa=pa, p=0.5, sigma0=1e-4, r0=1, c1=c, s1=s,
    #                           kappa_abs=kappa_abs, kappa_cont=kappa_cont, cont_weight=0.2)
    # model = [star, atg, atg2]
    # # model = [atg]
    # model_names = "_".join([m.shortname for m in model])
    #
    # plot_components(model, dim, pixel_size, wl, zoom=5,
    #                 savefig=test_dir / f"{model_names}.png", save_as_fits=False, norm=0.5)
    # plot_components(model, dim, pixel_size, wl, savefig=test_dir / f"{model_names}.fits", save_as_fits=True)
