import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import astropy.units as u
import numpy as np
from ppdmod import plot
from ppdmod.basic_components import assemble_components
from ppdmod.data import set_data
from ppdmod.fitting import compute_interferometric_chi_sq, compute_observables, \
    set_components_from_theta, run_fit, get_best_fit
from ppdmod.parameter import Parameter
from ppdmod.options import STANDARD_PARAMS, OPTIONS


DATA_DIR = Path("../data/fits/hd142527")
fits_files = list((DATA_DIR).glob("*HAWAII*fits"))
data = set_data(fits_files, wavelengths="all",
                fit_data=["vis2"], wavelength_range=[3., 3.3]*u.um)

pa = Parameter(**STANDARD_PARAMS.pa)
pa.value = 0.12*u.rad.to(u.deg)
pa.set(min=-45, max=45)
pa.free = True

inc = Parameter(**STANDARD_PARAMS.inc)
inc.value = 0.83
inc.free = True

fc = Parameter(**STANDARD_PARAMS.fr)
fc.value = 0.56
fc.free = True

wl, flux_ratio = np.load(Path("../data/flux/hd142527/hd142527_flux_ratio.npy"))
fs = Parameter(**STANDARD_PARAMS.fr)
fs.value, fs.grid = flux_ratio, wl
fs.interpolate, fs.free = True, False

wl, k = np.load(Path("../data/flux/hd142527/hd142527_slope.npy"))
ks = Parameter(**STANDARD_PARAMS.exp)
ks.value, ks.grid = k, wl
ks.interpolate, ks.free = True, False

kc = Parameter(**STANDARD_PARAMS.exp)
kc.value = -3.9
kc.set(min=-20, max=20)

la = Parameter(**STANDARD_PARAMS.la)
la.value = 0.06
la.set(min=-1, max=1.5)

flor = Parameter(**STANDARD_PARAMS.fr)
flor.value = 0.43
flor.free = True

params = {"fc": fc, "flor": flor, "la": la, "kc": kc, "inc": inc, "pa": pa}
labels = [label for label in params]

OPTIONS.model.constant_params = {"ks": 1, "fs": fs}
OPTIONS.model.components_and_params = [["StarHaloGaussLor", params]]
OPTIONS.fit.method = "dynesty"

result_dir = Path("results/matisse")
result_dir.mkdir(exist_ok=True, parents=True)
model_name = "starHaloGaussLor"

components = assemble_components(
        OPTIONS.model.components_and_params,
        OPTIONS.model.shared_params)

rchi_sq = compute_interferometric_chi_sq(
        *compute_observables(components), reduced=True)
print(f"rchi_sq: {rchi_sq}")

plot.plot_overview(savefig=result_dir / f"{model_name}_data_overview.pdf")
plot.plot_fit(components[0].inc(), components[0].pa(), components=components,
              savefig=result_dir / f"{model_name}_pre_fit_results.pdf")


if __name__ == "__main__":
    ncores = None
    fit_params_emcee = {"nburnin": 2000, "nsteps": 8000, "nwalkers": 100}
    fit_params_dynesty = {"nlive": 1500, "sample": "rwalk", "bound": "multi"}

    if OPTIONS.fit.method == "emcee":
        fit_params = fit_params_emcee
        ncores = fit_params["nwalkers"]//2 if ncores is None else ncores
        fit_params["discard"] = fit_params["nburnin"]
    else:
        ncores = 50 if ncores is None else ncores
        fit_params = fit_params_dynesty
    
    sampler = run_fit(**fit_params, ncores=ncores,
                      save_dir=result_dir, debug=False)

    theta, uncertainties = get_best_fit(
            sampler, **fit_params, method="quantile")

    components_and_params, shared_params = set_components_from_theta(theta)
    components = assemble_components(components_and_params, shared_params)
    rchi_sq = compute_interferometric_chi_sq(
            *compute_observables(components), reduced=True)
    print(f"rchi_sq: {rchi_sq}")

    plot.plot_chains(sampler, labels, **fit_params,
                     savefig=result_dir / f"{model_name}_{OPTIONS.fit.method}_chains.pdf")
    plot.plot_corner(sampler, labels, **fit_params,
                     savefig=result_dir / f"{model_name}_{OPTIONS.fit.method}_corner.pdf")
    plot.plot_fit(components[0].inc(), components[0].pa(), components=components,
                  savefig=result_dir / f"{model_name}_{OPTIONS.fit.method}_fit_results.pdf")
    plot.plot_components(components, 512, 0.02, 3.14, savefig=result_dir / f"{model_name}_{OPTIONS.fit.method}_image.pdf")
