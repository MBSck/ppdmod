from datetime import datetime
from pathlib import Path

import astropy.units as u
import numpy as np

from ppdmod import analysis
from ppdmod import custom_components
from ppdmod import data
from ppdmod import fitting
from ppdmod import plot
from ppdmod import utils
from ppdmod.parameter import Parameter
from ppdmod.options import STANDARD_PARAMETERS, OPTIONS


DATA_DIR = Path("tests/data")

OPTIONS.fit.data = ["flux", "vis2", "t3"]
# wavelengths = [1.6]*u.um
# wavelengths = [2.25]*u.um
# wavelengths = [1.6, 2.25]*u.um
# wavelengths = [1.6, 2.25, 3.5]*u.um
# wavelengths = [3.5]*u.um
# wavelengths = [1.6, 2.25, 3.5, 8., 9., 10., 11.3, 12.5]*u.um
wavelengths = [8., 9., 10., 11.3, 12.5]*u.um
data.set_fit_wavelengths(wavelengths)
fits_files = list((DATA_DIR / "fits").glob("*04-23*AQU*fits"))
data.set_data(fits_files)
data.set_fit_weights()

wavelength_axes = list(
    map(lambda x: x.wavelength, OPTIONS.data.readouts))
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

wl_dhs, opacity_dhs = utils.load_data(
        qval_files, load_func=utils.qval_to_opacity)
wl_grf, opacity_grf = utils.load_data(grf_files)
wl_op, opacity = wl_grf, opacity_grf

opacity = utils.linearly_combine_data(opacity, weights)
opacity = np.interp(wavelength_axes.value, wl_op[0], opacity)

# wl_cont, cont_opacity = utils.load_data(DATA_DIR / "qval" / "Q_amorph_c_rv0.1.dat",
#                                         load_func=utils.qval_to_opacity)
wl_cont, cont_opacity = utils.load_data(DATA_DIR / "qval" / "Q_iron_0.10um_dhs_0.99.dat",
                                        load_func=utils.qval_to_opacity)
cont_opacity = np.interp(wavelength_axes.value, wl_cont, cont_opacity)

kappa_abs = Parameter(**STANDARD_PARAMETERS["kappa_abs"])
kappa_abs.value, kappa_abs.wavelength = opacity, wavelength_axes
kappa_cont = Parameter(**STANDARD_PARAMETERS["kappa_cont"])
kappa_cont.value, kappa_cont.wavelength = cont_opacity, wavelength_axes

dim, distance = 32, 148.3
OPTIONS.model.constant_params = {
    "dim": dim, "dist": distance,
    "eff_temp": 7500, "f": star_flux,
    "pa": 162, "elong": 0.56,
    "eff_radius": 1.75, "kappa_abs": kappa_abs,
    "kappa_cont": kappa_cont}

rin = Parameter(**STANDARD_PARAMETERS["rin"])
rout = Parameter(**STANDARD_PARAMETERS["rout"])
p = Parameter(**STANDARD_PARAMETERS["p"])
inner_sigma = Parameter(**STANDARD_PARAMETERS["inner_sigma"])

rin.value = 1.
rout.value = 2.
p.value = 0.5
inner_sigma.value = 1e-3

rin.set(min=0.5, max=5)
rout.set(min=1.5, max=6)
p.set(min=0., max=1.)
inner_sigma.set(min=0, max=1e-2)

rout.free = True

inner_ring = {"rin": rin, "rout": rout, "inner_sigma": inner_sigma, "p": p}
inner_ring_labels = [f"ir_{label}" for label in inner_ring]

rin = Parameter(**STANDARD_PARAMETERS["rin"])
rout = Parameter(**STANDARD_PARAMETERS["rout"])
a = Parameter(**STANDARD_PARAMETERS["a"])
phi = Parameter(**STANDARD_PARAMETERS["phi"])
p = Parameter(**STANDARD_PARAMETERS["p"])
inner_sigma = Parameter(**STANDARD_PARAMETERS["inner_sigma"])

rin.value = 13
a.value = 0.5
phi.value = 130
p.value = 0.5
inner_sigma.value = 1e-3

# NOTE: Set outer radius to be constant and calculate flux once?
rin.set(min=1, max=40)
p.set(min=0., max=1.)
inner_sigma.set(min=0, max=1e-2)
a.set(min=0., max=1.)
phi.set(min=0, max=360)

rout.free = True

outer_ring = {"rin": rin, "a": a, "phi": phi, "inner_sigma": inner_sigma, "p": p}
outer_ring_labels = [f"or_{label}" for label in outer_ring]

# q = Parameter(**STANDARD_PARAMETERS["q"])
# inner_temp = Parameter(**STANDARD_PARAMETERS["inner_temp"])
pa = Parameter(**STANDARD_PARAMETERS["pa"])
elong = Parameter(**STANDARD_PARAMETERS["elong"])
cont_weight = Parameter(**STANDARD_PARAMETERS["cont_weight"])

# q.value = 0.5
# inner_temp.value = 1500
pa.value = 163
elong.value = 0.5
cont_weight.value = 0.54             # Relative contribution (adds to 1). Mass fractions

# q.set(min=0., max=1.)
# inner_temp.set(min=300, max=2000)
pa.set(min=0, max=360)
elong.set(min=0, max=1)
cont_weight.set(min=0.3, max=0.8)

# OPTIONS.model.shared_params = {"q": q, "inner_temp": inner_temp,
#                                "pa": pa, "elong": elong,
#                                "cont_weight": cont_weight}
OPTIONS.model.shared_params = {"pa": pa, "elong": elong,
                               "cont_weight": cont_weight}
# OPTIONS.model.shared_params = {"cont_weight": cont_weight}
shared_params_labels = [f"sh_{label}" for label in OPTIONS.model.shared_params]

OPTIONS.model.components_and_params = [
    ["Star", {}],
    # ["GreyBody", inner_ring],
    ["AsymmetricGreyBody", outer_ring],
]

# labels = inner_ring_labels + outer_ring_labels + shared_params_labels
# labels = inner_ring_labels + shared_params_labels
labels = outer_ring_labels + shared_params_labels

# component_labels = ["Star", "Inner Ring", "Outer Ring"]
# component_labels = ["Star", "Inner Ring"]
component_labels = ["Star", "Outer Ring"]

OPTIONS.model.modulation = 1
OPTIONS.model.gridtype = "logarithmic"
OPTIONS.fit.method = "dynesty"

model_result_dir = Path("../model_results/")
day_dir = model_result_dir / str(datetime.now().date())
dir_name = f"results_model_{datetime.now().strftime('%H:%M:%S')}"
result_dir = day_dir / dir_name
result_dir.mkdir(parents=True, exist_ok=True)

pre_fit_dir = result_dir / "pre_fit"
pre_fit_dir.mkdir(parents=True, exist_ok=True)

components = custom_components.assemble_components(
        OPTIONS.model.components_and_params,
        OPTIONS.model.shared_params)


plot.plot_overview(savefig=pre_fit_dir / "data_overview.pdf")
plot.plot_observables("hd142666", [3, 12]*u.um, components,
                      save_dir=pre_fit_dir)

analysis.save_fits(
        4096, 0.1, distance,
        components, component_labels,
        opacities=[kappa_abs, kappa_cont],
        savefits=pre_fit_dir / "model.fits",
        object_name="HD 142666")

post_fit_dir = result_dir / "post_fit"
post_fit_dir.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ncores = None
    fit_params_emcee = {"nburnin": 200, "nsteps": 500, "nwalkers": 100}
    fit_params_dynesty = {"nlive": 1500, "sample": "rwalk", "bound": "multi"}

    if OPTIONS.fit.method == "emcee":
        fit_params = fit_params_emcee
        ncores = fit_params["nwalkers"]//2 if ncores is None else ncores
        fit_params["discard"] = fit_params["nburnin"]
    else:
        ncores = 30 if ncores is None else ncores
        fit_params = fit_params_dynesty

    sampler = fitting.run_fit(
            **fit_params, ncores=ncores,
            save_dir=post_fit_dir, debug=False)

    theta, uncertainties = fitting.get_best_fit(
            sampler, **fit_params, method="quantile")

    plot.plot_chains(sampler, labels, **fit_params,
                     savefig=post_fit_dir / "chains.pdf")
    plot.plot_corner(sampler, labels, **fit_params,
                     savefig=post_fit_dir / "corner.pdf")
    new_params = dict(zip(labels, theta))

    components_and_params, shared_params = fitting.set_params_from_theta(theta)
    components = custom_components.assemble_components(
            components_and_params, shared_params)

    plot.plot_observables("hd142666", [3, 12]*u.um, components,
                          save_dir=post_fit_dir)

    analysis.save_fits(
            4096, 0.1, distance,
            components, component_labels,
            opacities=[kappa_abs, kappa_cont],
            savefits=post_fit_dir / "model.fits",
            object_name="HD 142666", **fit_params, ncores=ncores)

    # axis_ratio = OPTIONS.model.constant_params["elong"]
    # pos_angle = OPTIONS.model.constant_params["pa"]
    axis_ratio = shared_params["elong"]
    compression = shared_params["pa"]

    plot.plot_fit(
            axis_ratio, compression,
            components=components,
            savefig=post_fit_dir / "fit_results.pdf")
