import os
from typing import List
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import astropy.units as u
from ppdmod import plot
from ppdmod.basic_components import assemble_components
from ppdmod.data import set_data
from ppdmod.fitting import compute_observable_chi_sq, compute_observables, \
    set_params_from_theta, run_fit, get_best_fit
from ppdmod.parameter import Parameter
from ppdmod.options import STANDARD_PARAMETERS, OPTIONS


DATA_DIR = Path("../data/aspro/")
OPTIONS.model.output = "non-physical"
fits_files = [DATA_DIR / "a1_phi0_rin2_modulated_ring.fits"]
data = set_data(fits_files, wavelengths=[3.5]*u.um, fit_data=["vis", "t3"])

rin = Parameter(**STANDARD_PARAMETERS.rin)
rin.value = 2

a = Parameter(**STANDARD_PARAMETERS.a)
a.value = 1

phi = Parameter(**STANDARD_PARAMETERS.phi)
phi.value = 0

params = {"rin": rin, "a": a, "phi": phi}
labels = [label for label in params]

model_name = "Ring"
OPTIONS.model.components_and_params = [[model_name, params]]
OPTIONS.model.gridtype = "logarithmic"
OPTIONS.fit.method = "dynesty"

result_dir = Path("results/aspro")
result_dir.mkdir(exist_ok=True, parents=True)

components = assemble_components(
        OPTIONS.model.components_and_params,
        OPTIONS.model.shared_params)

rchi_sq = compute_observable_chi_sq(
        *compute_observables(components), reduced=True)
print(f"rchi_sq: {rchi_sq}")

plot.plot_overview(savefig=result_dir / f"{model_name}_data_overview.pdf")
plot.plot_fit(components[0].inc(), components[0].pa(), components=components,
              savefig=result_dir / f"{model_name}_pre_fit_results.pdf")


if __name__ == "__main__":
    ncores = None
    fit_params_emcee = {"nburnin": 2000, "nsteps": 8000, "nwalkers": 100}
    fit_params_dynesty = {"nlive": 2000, "sample": "rwalk", "bound": "multi"}

    if OPTIONS.fit.method == "emcee":
        fit_params = fit_params_emcee
        ncores = fit_params["nwalkers"]//2 if ncores is None else ncores
        fit_params["discard"] = fit_params["nburnin"]
    else:
        ncores = 50 if ncores is None else ncores
        fit_params = fit_params_dynesty

    sampler = run_fit(**fit_params, ncores=ncores, save_dir=result_dir, debug=False)
    theta, uncertainties = get_best_fit(
            sampler, **fit_params, method="quantile")

    components_and_params, shared_params = set_params_from_theta(theta)
    components = assemble_components(components_and_params, shared_params)
    rchi_sq = compute_observable_chi_sq(
            *compute_observables(components), reduced=True)
    print(f"rchi_sq: {rchi_sq}")

    plot.plot_chains(sampler, labels, **fit_params,
                     savefig=result_dir / f"{model_name}_{OPTIONS.fit.method}_chains.pdf")
    plot.plot_corner(sampler, labels, **fit_params,
                     savefig=result_dir / f"{model_name}_{OPTIONS.fit.method}_corner.pdf")
    plot.plot_fit(components[0].inc(), components[0].pa(), components=components,
                  savefig=result_dir / f"{model_name}_{OPTIONS.fit.method}_fit_results.pdf")
    plot.plot_components(components, 512, 0.02, 1.68, savefig=result_dir / f"{model_name}_{OPTIONS.fit.method}_image.pdf")
