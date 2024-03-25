from datetime import datetime
from pathlib import Path

import astropy.units as u
import numpy as np

# from ppdmod import analysis
from ppdmod import basic_components
from ppdmod import fitting
from ppdmod import plot
from ppdmod import utils
from ppdmod.data import set_data, get_all_wavelengths
from ppdmod.parameter import Parameter
from ppdmod.options import STANDARD_PARAMETERS, OPTIONS


DATA_DIR = Path("../tests/data")
wavelengths = {"hband": [1.6]*u.um,
               "kband": [2.25]*u.um,
               "lband": [3.2]*u.um,
               "nband": [8., 9., 10., 11.3, 12.5]*u.um}

fits_files = list((DATA_DIR / "fits" / "hd142666").glob("*fits"))
data = set_data(fits_files, wavelengths=wavelengths["nband"], fit_data=["vis2"])
wavelengths = get_all_wavelengths()

# TODO: Check flux values -> gave nan for only N-band
wl_flux, flux = utils.load_data(DATA_DIR / "flux/HD142666_stellar_model.txt.gz")
flux_interpn = np.interp(wavelengths.value, wl_flux, flux)
star_flux = Parameter(**STANDARD_PARAMETERS.f)
star_flux.value, star_flux.wavelength = flux_interpn, wavelengths

wl_flux_ratio, flux_ratio = np.load(DATA_DIR / "flux" / "flux_ratio_inner_disk_hd142666.npy")
flux_ratio_interpn = np.interp(wavelengths.value, wl_flux_ratio, flux_ratio)
flux_ratio_interpn += 0.2
point_flux_ratio = Parameter(**STANDARD_PARAMETERS.fr)
point_flux_ratio.value, point_flux_ratio.wavelength = flux_ratio_interpn, wavelengths

weights = np.array([42.8, 9.7, 43.5, 1.1, 2.3, 0.6])/100
names = ["olivine", "pyroxene", "forsterite", "enstatite"]
fmaxs = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
sizes = [[0.1, 1.5], [1.5], [0.1, 1.5], [1.5]]

roy_opacity = utils.get_opacity(DATA_DIR, weights, sizes, names, "qval",
                                wavelengths.value, fmaxs)

# TODO: Finish this for the Juhasz opacities and also check Roy's paper again
weights = np.array([42.8, 9.7, 43.5, 1.1, 2.3, 0.6])/100
names = ["olivine", "pyroxene", "forsterite", "enstatite"]
fmaxs = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
sizes = [[0.1, 1.5], [1.5], [0.1, 1.5], [1.5]]

juhasz_opacity = utils.get_opacity(DATA_DIR, weights, sizes, names, "qval",
                                   wavelengths.value, fmaxs)

opacity = roy_opacity

cont_opacity_file = DATA_DIR / "qval" / "Q_amorph_c_rv0.1.dat"
# cont_opacity_file = DATA_DIR / "qval" / "Q_iron_0.10um_dhs_0.7.dat",
wl_cont, cont_opacity = utils.load_data(cont_opacity_file, load_func=utils.qval_to_opacity)
cont_opacity = np.interp(wavelengths.value, wl_cont, cont_opacity)

kappa_abs = Parameter(**STANDARD_PARAMETERS.kappa_abs)
kappa_abs.value, kappa_abs.wavelength = opacity, wavelengths
kappa_cont = Parameter(**STANDARD_PARAMETERS.kappa_cont)
kappa_cont.value, kappa_cont.wavelength = cont_opacity, wavelengths

# TODO: Think of a better way to assign f than through const_params
dim, distance = 32, 148.3
OPTIONS.model.constant_params = {
    "dim": dim, "dist": distance,
    "eff_temp": 7500, "f": star_flux,
    # "fr": point_flux_ratio,
    # "pa": 162, "inc": 0.56,
    "eff_radius": 1.75, "kappa_abs": kappa_abs,
    "kappa_cont": kappa_cont}

x = Parameter(**STANDARD_PARAMETERS.x)
y = Parameter(**STANDARD_PARAMETERS.y)
x.value = 1
y.value = 0
x.set(min=-10, max=10)
y.set(min=-10, max=10)
x.free = y.free = True
point_source = {"x": x, "y": y}
point_source_labels = [f"pt_{label}" for label in point_source]

rin = Parameter(**STANDARD_PARAMETERS.rin)
rout = Parameter(**STANDARD_PARAMETERS.rout)
p = Parameter(**STANDARD_PARAMETERS.p)
inner_sigma = Parameter(**STANDARD_PARAMETERS.inner_sigma)

rin.value = 1.
rout.value = 2.
p.value = 0.5
inner_sigma.value = 1e-3

rin.set(min=0.5, max=5)
rout.set(min=1.5, max=6)
p.set(min=0., max=1.)
inner_sigma.set(min=0, max=1e-2)

rout.free = True

# inner_ring = {"x": x, "y": y, "rin": rin,
#               "rout": rout, "inner_sigma": inner_sigma, "p": p}
inner_ring = {"rin": rin, "rout": rout, "inner_sigma": inner_sigma, "p": p}
inner_ring_labels = [f"ir_{label}" for label in inner_ring]

rin = Parameter(**STANDARD_PARAMETERS.rin)
rout = Parameter(**STANDARD_PARAMETERS.rout)
p = Parameter(**STANDARD_PARAMETERS.p)
inner_sigma = Parameter(**STANDARD_PARAMETERS.inner_sigma)
a = Parameter(**STANDARD_PARAMETERS.a)
phi = Parameter(**STANDARD_PARAMETERS.phi)

rin.value = 13
p.value = 0.5
inner_sigma.value = 1e-3
a.value = 0.5
phi.value = 130

# NOTE: Set outer radius to be constant and calculate flux once?
rin.set(min=1, max=40)
p.set(min=0., max=1.)
inner_sigma.set(min=0, max=1e-2)
a.set(min=0., max=1.)
phi.set(min=0, max=180)

rout.free = True

outer_ring = {"rin": rin, "a": a, "phi": phi, "inner_sigma": inner_sigma, "p": p}
# outer_ring = {"rin": rin, "inner_sigma": inner_sigma, "p": p}
outer_ring_labels = [f"or_{label}" for label in outer_ring]

# q = Parameter(**STANDARD_PARAMETERS.q)
# inner_temp = Parameter(**STANDARD_PARAMETERS.inner_temp)
pa = Parameter(**STANDARD_PARAMETERS.pa)
inc = Parameter(**STANDARD_PARAMETERS.inc)
cont_weight = Parameter(**STANDARD_PARAMETERS.cont_weight)

# q.value = 0.5
# inner_temp.value = 1500
pa.value = 163
inc.value = 0.5
cont_weight.value = 0.40             # Relative contribution (adds to 1). Mass fractions

# q.set(min=0., max=1.)
# inner_temp.set(min=300, max=2000)
pa.set(min=0, max=180)
inc.set(min=0.3, max=0.95)
cont_weight.set(min=0.3, max=0.8)

# OPTIONS.model.shared_params = {"q": q, "inner_temp": inner_temp,
#                                "pa": pa, "inc": inc,
#                                "cont_weight": cont_weight}
OPTIONS.model.shared_params = {"pa": pa, "inc": inc,
                               "cont_weight": cont_weight}
# OPTIONS.model.shared_params = {"cont_weight": cont_weight}
shared_params_labels = [f"sh_{label}" for label in OPTIONS.model.shared_params]

OPTIONS.model.components_and_params = [
    # ["PointSource", point_source],
    ["Star", {}],
    ["GreyBody", inner_ring],
    # ["GreyBody", outer_ring],
    # ["AsymmetricGreyBody", outer_ring],
]

# labels = inner_ring_labels + outer_ring_labels + shared_params_labels
labels = inner_ring_labels + shared_params_labels
# labels = point_source_labels + outer_ring_labels + shared_params_labels
# labels = outer_ring_labels + shared_params_labels

# component_labels = ["Star", "Inner Ring", "Outer Ring"]
# component_labels = ["Star", "Inner Ring"]
# component_labels = ["Point Source", "Outer Ring"]
# component_labels = ["star", "Outer ring"]

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

components = basic_components.assemble_components(
        OPTIONS.model.components_and_params,
        OPTIONS.model.shared_params)

plot.plot_overview(savefig=pre_fit_dir / "data_overview.pdf")
# plot.plot_observables("hd142666", [1, 12]*u.um, components,
#                       save_dir=pre_fit_dir)

# analysis.save_fits(
#         4096, 0.1, distance,
#         components, component_labels,
#         opacities=[kappa_abs, kappa_cont],
#         savefits=pre_fit_dir / "model.fits",
#         object_name="HD 142666")

post_fit_dir = result_dir / "post_fit"
post_fit_dir.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ncores = 6
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
            save_dir=post_fit_dir, debug=True)

    theta, uncertainties = fitting.get_best_fit(
            sampler, **fit_params, method="quantile")

    plot.plot_chains(sampler, labels, **fit_params,
                     savefig=post_fit_dir / "chains.pdf")
    plot.plot_corner(sampler, labels, **fit_params,
                     savefig=post_fit_dir / "corner.pdf")
    new_params = dict(zip(labels, theta))

    components_and_params, shared_params = fitting.set_params_from_theta(theta)
    components = basic_components.assemble_components(
            components_and_params, shared_params)

    # plot.plot_observables("hd142666", [1, 12]*u.um, components,
    #                       save_dir=post_fit_dir)

    # analysis.save_fits(
    #         4096, 0.1, distance,
    #         components, component_labels,
    #         opacities=[kappa_abs, kappa_cont],
    #         savefits=post_fit_dir / "model.fits",
    #         object_name="HD 142666", **fit_params, ncores=ncores)

    # axis_ratio = OPTIONS.model.constant_params["inc"]
    # pos_angle = OPTIONS.model.constant_params["pa"]
    inclination = shared_params["inc"]
    pos_angle = shared_params["pa"]

    plot.plot_fit(
            inclination, pos_angle,
            components=components,
            savefig=post_fit_dir / "fit_results.pdf")
