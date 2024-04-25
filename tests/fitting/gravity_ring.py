from typing import List
from pathlib import Path

import astropy.units as u
import numpy as np
from ppdmod import data
from ppdmod import plot
from ppdmod.basic_components import assemble_components
from ppdmod.fitting import compute_observable_chi_sq, compute_observables, \
    set_params_from_theta, lnprior, run_fit, get_best_fit, transform_uniform_prior
from ppdmod.parameter import Parameter
from ppdmod.options import STANDARD_PARAMETERS, OPTIONS
from ppdmod.utils import compute_photometric_slope


def ptform(theta: List[float]) -> np.ndarray:
    """Transform that constrains the first two parameters to 1."""
    params = transform_uniform_prior(theta)
    params[1] = params[1]*(1-params[0])
    return params


def lnprob(theta: np.ndarray) -> float:
    """Takes theta vector and the x, y and the yerr of the theta.
    Returns a number corresponding to how good of a fit the model is to your
    data for a given set of parameters, weighted by the data points.

    This is the analytical 1D implementation.

    Constraints the parameters to 1 for emcee.

    Parameters
    ----------
    theta: np.ndarray
        The parameters that ought to be fitted.

    Returns
    -------
    float
        The log of the probability.
    """
    parameters, shared_params = set_params_from_theta(theta)
    if parameters[0][1]["fs"] + parameters[0][1]["fc"] > 1:
        return -np.inf

    if OPTIONS.fit.method == "emcee":
        if np.isinf(lnprior(parameters, shared_params)):
            return -np.inf

    components = assemble_components(parameters, shared_params)
    return compute_observable_chi_sq(*compute_observables(components))


DATA_DIR = Path("../data/gravity")
OPTIONS.model.output = "non-physical"
fits_files = list((DATA_DIR).glob("*fits"))
data.set_data(fits_files, wavelengths="all", fit_data=["vis2", "t3"])

pa = Parameter(**STANDARD_PARAMETERS.pa)
pa.value = 14
pa.free = True

inc = Parameter(**STANDARD_PARAMETERS.inc)
inc.value = 0.91
inc.free = True

fc = Parameter(**STANDARD_PARAMETERS.fr)
fc.value = 0.66
fc.free = True

fs = Parameter(**STANDARD_PARAMETERS.fr)
fs.value = 0.34
fs.free = True

wavelength = data.get_all_wavelengths()
ks = Parameter(**STANDARD_PARAMETERS.exp)
ks.value = compute_photometric_slope(wavelength, 6500)
ks.wavelength = wavelength
ks.free = False

kc = Parameter(**STANDARD_PARAMETERS.exp)
kc.value = -1.47273589478313
kc.set(min=-10, max=10)

la = Parameter(**STANDARD_PARAMETERS.la)
la.value = 0.10037054511756291
la.set(min=-1, max=1.5)

lkr = Parameter(**STANDARD_PARAMETERS.lkr)
lkr.value = 
lkr.set(min=-1, max=1)

flor = Parameter(**STANDARD_PARAMETERS.fr)
flor.value = 1.
flor.free = True

a = Parameter(**STANDARD_PARAMETERS.a)
a.value = 

phi = Parameter(**STANDARD_PARAMETERS.phi)
phi.value = 

params = {"fs": fs, "fc": fc, "flor": flor, "la": la,
          "lkr": lkr, "kc": kc, "inc": inc, "pa": pa, "a": a, "phi": phi}
labels = [label for label in params]

OPTIONS.model.constant_params = {"wl0": 2.15, "ks": ks}
OPTIONS.model.components_and_params = [["StarHaloRing", params]]
OPTIONS.model.gridtype = "logarithmic"
OPTIONS.fit.method = "dynesty"

result_dir = Path("results/gravity")
result_dir.mkdir(exist_ok=True, parents=True)
model_name = "starHaloGaussLorRing"

plot.plot_overview(savefig=result_dir / f"{model_name}_data_overview.pdf")


if __name__ == "__main__":
    ncores = None
    fit_params_emcee = {"nburnin": 2000, "nsteps": 8000, "nwalkers": 100,
                        "lnprob": lnprob}
    fit_params_dynesty = {"nlive": 1500, "sample": "rwalk", "bound": "multi",
                          "ptform": ptform}

    if OPTIONS.fit.method == "emcee":
        fit_params = fit_params_emcee
        ncores = fit_params["nwalkers"]//2 if ncores is None else ncores
        fit_params["discard"] = fit_params["nburnin"]
    else:
        ncores = 50 if ncores is None else ncores
        fit_params = fit_params_dynesty

    sampler = run_fit(**fit_params, ncores=ncores, save_dir=result_dir, debug=False)
    theta, uncertainties = get_best_fit(sampler, **fit_params, method="quantile")

    components_and_params, shared_params = set_params_from_theta(theta)
    components = assemble_components(components_and_params, shared_params)
    rchi_sq = compute_observable_chi_sq(*compute_observables(components), reduced=True)
    print(f"rchi_sq: {rchi_sq}")

    plot.plot_chains(sampler, labels, **fit_params,
                     savefig=result_dir / f"{model_name}_{OPTIONS.fit.method}_chains.pdf")
    plot.plot_corner(sampler, labels, **fit_params,
                     savefig=result_dir / f"{model_name}_{OPTIONS.fit.method}_corner.pdf")

    plot.plot_fit(components[0].inc(), components[0].pa(), components=components,
                  savefig=result_dir / f"{model_name}_{OPTIONS.fit.method}_fit_results.pdf")
