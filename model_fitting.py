from datetime import datetime
from pathlib import Path

import astropy.units as u
import numpy as np

from ppdmod import custom_components
from ppdmod import data
from ppdmod import fitting
from ppdmod import plot
from ppdmod import utils
from ppdmod.parameter import STANDARD_PARAMETERS, Parameter
from ppdmod.options import OPTIONS


# TODO: Make function that saves model parameters to load.
DATA_DIR = Path("tests/data")

OPTIONS["fit.data"] = ["flux", "vis2", "t3phi"]
OPTIONS["data.binning.window"] = 0.1*u.um
wavelengths = [2.25]*u.um
# wavelengths = [1.6, 2.25, 3.5, 8., 9., 10., 11.3, 12.5]*u.um
data.set_fit_wavelengths(wavelengths)
fits_files = list((DATA_DIR / "fits").glob("*.fits"))
data.set_data(fits_files)

# TODO: Check if the configuration of these parameters is ok
wavelength_axes = list(
    map(lambda x: x.wavelength, OPTIONS["data.readouts"]))
wavelength_axes = np.sort(np.unique(np.concatenate(wavelength_axes)))

wl_flux, flux = utils.load_data(DATA_DIR / "flux/HD142666_stellar_model.txt.gz")
flux = np.interp(wavelength_axes.value, wl_flux, flux)
star_flux = Parameter(**STANDARD_PARAMETERS["f"])
star_flux.value, star_flux.wavelength = flux, wavelength_axes

weights = np.array([42.8, 9.7, 43.5, 1.1, 2.3, 0.6])/100
qval_files = ["Q_Am_Mgolivine_Jae_DHS_f1.0_rv0.1.dat",
              "Q_Am_Mgolivine_Jae_DHS_f1.0_rv1.5.dat",
              "Q_Am_Mgpyroxene_Dor_DHS_f1.0_rv1.5.dat",
              "Q_Fo_Suto_DHS_f1.0_rv0.1.dat",
              "Q_Fo_Suto_DHS_f1.0_rv1.5.dat",
              "Q_En_Jaeger_DHS_f1.0_rv1.5.dat"]
qval_files = list(map(lambda x: DATA_DIR / "qval" / x, qval_files))

grf_files = ["MgOlivine0.1.Combined.Kappa",
             "MgOlivine2.0.Combined.Kappa",
             "MgPyroxene2.0.Combined.Kappa",
             "Forsterite0.1.Combined.Kappa",
             "Forsterite2.0.Combined.Kappa",
             "Enstatite2.0.Combined.Kappa"]
grf_files = list(map(lambda x: DATA_DIR / "grf" / x, grf_files))

wl_dhs, opacity_dhs = utils.load_data(qval_files, load_func=utils.qval_to_opacity)
wl_grf, opacity_grf = utils.load_data(grf_files)
wl_op, opacity = wl_grf, opacity_grf

opacity = utils.linearly_combine_data(opacity, weights)
opacity = np.interp(wavelength_axes.value, wl_op[0], opacity)

wl_cont, cont_opacity = utils.load_data(DATA_DIR / "qval" / "Q_amorph_c_rv0.1.dat",
                                        load_func=utils.qval_to_opacity)
# wl_cont, cont_opacity = utils.load_data(qval_dir / "Q_iron_0.10um_dhs_0.99.dat",
#                                         load_func=utils.qval_to_opacity)
cont_opacity = np.interp(wavelength_axes.value, wl_cont, cont_opacity)

kappa_abs = Parameter(**STANDARD_PARAMETERS["kappa_abs"])
kappa_abs.value, kappa_abs.wavelength = opacity, wavelength_axes
kappa_cont = Parameter(**STANDARD_PARAMETERS["kappa_cont"])
kappa_cont.value, kappa_cont.wavelength = cont_opacity, wavelength_axes

dim, distance = 32, 148.3
OPTIONS["model.constant_params"] = {
    "dim": dim, "dist": 148.3,
    "eff_temp": 7500, "f": star_flux,
    "eff_radius": 1.75, "inner_temp": 1500,
    "kappa_abs": kappa_abs, "kappa_cont": kappa_cont}

rin = Parameter(**STANDARD_PARAMETERS["rin"])
rout = Parameter(**STANDARD_PARAMETERS["rout"])
p = Parameter(**STANDARD_PARAMETERS["p"])
inner_sigma = Parameter(**STANDARD_PARAMETERS["inner_sigma"])

rin.value = 1.
rout.value = 7.
p.value = 0.5
inner_sigma.value = 1e-3

rin.set(min=0.5, max=3)
rout.set(min=1., max=4)
p.set(min=0., max=1.)
inner_sigma.set(min=0, max=1e-2)

rout.free = True

inner_ring = {"rin": rin, "rout": rout, "inner_sigma": inner_sigma, "p": p}
inner_ring_labels = [f"ir_{label}" for label in inner_ring]

# rin = Parameter(**STANDARD_PARAMETERS["rin"])
# rout = Parameter(**STANDARD_PARAMETERS["rout"])
# a = Parameter(**STANDARD_PARAMETERS["a"])
# phi = Parameter(**STANDARD_PARAMETERS["phi"])
# p = Parameter(**STANDARD_PARAMETERS["p"])
# inner_sigma = Parameter(**STANDARD_PARAMETERS["inner_sigma"])

# rin.value = 13
# a.value = 0.5
# phi.value = 130
# p.value = 0.5
# inner_sigma.value = 1e-3

# # NOTE: Set outer radius to be constant and calculate flux once?
# rin.set(min=4, max=30)
# p.set(min=0., max=1.)
# inner_sigma.set(min=0, max=1e-2)
# a.set(min=0., max=1.)
# phi.set(min=0, max=360)

# rout.free = True

# outer_ring = {"rin": rin, "a": a, "phi": phi, "inner_sigma": inner_sigma, "p": p}
# outer_ring_labels = [f"or_{label}" for label in outer_ring]

q = Parameter(**STANDARD_PARAMETERS["q"])
pa = Parameter(**STANDARD_PARAMETERS["pa"])
elong = Parameter(**STANDARD_PARAMETERS["elong"])
cont_weight = Parameter(**STANDARD_PARAMETERS["cont_weight"])

q.value = 0.5
pa.value = 145
elong.value = 0.5
cont_weight.value = 130

q.set(min=0., max=1.)
pa.set(min=0, max=360)
elong.set(min=0, max=1)
cont_weight.set(min=0., max=1.)

OPTIONS["model.shared_params"] = {"q": q, "pa": pa, "elong": elong,
                                  "cont_weight": cont_weight}
shared_params_labels = [f"sh_{label}"
                        for label in OPTIONS["model.shared_params"]]

OPTIONS["model.components_and_params"] = [
    ["Star", {}],
    ["AnalyticalTempGradient", inner_ring],
    # ["AnalyticalAsymmetricTempGradient", outer_ring],
]

# labels = inner_ring_labels + outer_ring_labels + shared_params_labels
labels = inner_ring_labels + shared_params_labels

OPTIONS["model.modulation.order"] = 1
OPTIONS["model.gridtype"] = "logarithmic"

OPTIONS["fit.method"] = "emcee"

model_result_dir = Path("../model_results/")
day_dir = model_result_dir / str(datetime.now().date())
time = datetime.now()
file_name = f"results_model_{time.hour}:{time.minute}:{time.second}"
result_dir = day_dir / file_name
if not result_dir.exists():
    result_dir.mkdir(parents=True)

if __name__ == "__main__":
    nburnin, nsteps, nwalkers = 250, 500, 100
    ncores = nwalkers // 2
    # ncores = 6
    sampler = fitting.run_fit(
        nwalkers=nwalkers, nsteps=nsteps,
        nburnin=nburnin, ncores=ncores, debug=False)
    theta = fitting.get_best_fit(sampler, discard=nburnin)

    plot.plot_chains(sampler, labels, discard=nburnin,
                     savefig=result_dir / "chains.pdf")
    plot.plot_corner(sampler, labels, discard=nburnin,
                     savefig=result_dir / "corner.pdf")
    np.save(result_dir / "best_fit_params.npy", theta)
    # result_dir = Path("/Users/scheuck/Data/model_results/2023-11-23/results_model_nsteps7500_nwalkers100_11:18:52")
    # theta = np.load(result_dir / "best_fit_params.npy")
    new_params = dict(zip(labels, theta))

    components_and_params, shared_params = fitting.set_params_from_theta(theta)
    components = custom_components.assemble_components(
            components_and_params, shared_params)
    component_labels = ["Star", "Inner Ring", "Outer Ring"]

    # HACK: This is to include innermost radius for rn.
    innermost_radius = components[1].params["rin"]
    for component in components:
        component.params["rin0"] = innermost_radius

    # plot.save_fits(
    #         4096, 0.1, distance,
    #         OPTIONS["fit.wavelengths"], components,
    #         component_labels, opacities=[kappa_abs, kappa_cont],
    #         savefits=result_dir / "model.fits",
    #         options=OPTIONS, object_name="HD 142666",
    #         nwalkers=nwalkers, nsteps=nburnin+nsteps,
    #         ncores=ncores)

    plot.plot_fit(
            new_params["sh_elong"], new_params["sh_pa"],
            components=components,
            savefig=result_dir / "fit_results.pdf")
