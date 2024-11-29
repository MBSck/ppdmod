from pathlib import Path

from ppdmod.basic_components import Ring
from ppdmod.data import set_data
from ppdmod.options import OPTIONS
from ppdmod.plot import plot_components


def image_ring(save_dir: Path) -> None:
    OPTIONS.model.modulation = 1
    modulation_amps = [(1, 0), (0, 0), (0, 0)]
    modulation_amps = modulation_amps[: OPTIONS.model.modulation]
    modulation_dict = {f"c{i+1}": amp[0] for i, amp in enumerate(modulation_amps)}
    modulation_dict.update({f"s{i+1}": amp[1] for i, amp in enumerate(modulation_amps)})

    dim, pixel_size, wl = 4096, 0.02, 3.5
    params = {"rin": 2, "width": 0.5, "cinc": 1, "pa": 0}
    thin, asymmetric = False, False
    ring = Ring(
        has_outer_radius=False,
        asymmetric=asymmetric,
        thin=thin,
        **params,
        **modulation_dict,
    )

    param_labels = [f"{key}{value}" for key, value in params.items()]
    if asymmetric:
        param_labels.extend([f"{key}{value}" for key, value in modulation_dict.items()])

    save_name = "_".join([label for label in [ring.name, *param_labels] if label]).replace(".", "")

    plot_components(
        [ring],
        dim,
        pixel_size,
        wl,
        zoom=5,
        savefig=save_dir / f"{save_name}.png",
        save_as_fits=False,
        norm=1,
    )

    plot_components(
        [ring],
        dim,
        pixel_size,
        wl,
        savefig=save_dir / f"{save_name}.fits",
        save_as_fits=True,
    )

    OPTIONS.model.modulation = 1


if __name__ == "__main__":
    fitting_dir = Path(__file__).parent.parent / "data" / "fits" / "hd142527"
    fits_files = list((fitting_dir).glob("*_L_*.fits"))
    data = set_data(fits_files, wavelengths="all", fit_data=["vis2", "t3"])

    image_dir = Path("images/")
    image_dir.mkdir(parents=True, exist_ok=True)
    image_ring(image_dir / "ring")

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
    # model_names = "_".join([m.name for m in model])
    #
    # plot_components(model, dim, pixel_size, wl, zoom=5,
    #                 savefig=test_dir / f"{model_names}.png", save_as_fits=False, norm=0.5)
    # plot_components(model, dim, pixel_size, wl, savefig=test_dir / f"{model_names}.fits", save_as_fits=True)
