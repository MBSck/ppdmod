from pathlib import Path

import astropy.units as u
import astropy.constants as const
import numpy as np
from astropy.modeling.models import BlackBody
from ppdmod import data
from ppdmod import fitting
from ppdmod import plot
from ppdmod.basic_components import assemble_components
from ppdmod.parameter import Parameter
from ppdmod.options import STANDARD_PARAMETERS, OPTIONS


DATA_DIR = Path("../data/gravity")
OPTIONS.model.output = "non-physical"
fits_files = list((DATA_DIR).glob("*fits"))
data.set_data(fits_files, wavelengths="all", fit_data=["vis2", "t3"])

pa = Parameter(**STANDARD_PARAMETERS.pa)
pa.value = 162
pa.free = True

inc = Parameter(**STANDARD_PARAMETERS.inc)
inc.value = 0.5
inc.free = True

fc = Parameter(**STANDARD_PARAMETERS.fr)
fc.value = 0.5
fc.free = True

fs = Parameter(**STANDARD_PARAMETERS.fr)
fs.value = 0.4
fs.free = True

# TODO: Check the rate of increase 0.05 again
wavelength = data.get_all_wavelengths()
wavelength = np.append(wavelength.copy(), (wavelength[-1].value+0.05104995)*u.um)
bb = np.log(BlackBody(temperature=7500*u.K)(wavelength).value)
freq = np.log((const.c / wavelength.to(u.m)).to(u.Hz).value)
dbb, dfreq = np.diff(bb), np.diff(freq)
ks = Parameter(**STANDARD_PARAMETERS.exp)
ks.value = dbb/dfreq
ks.wavelength = wavelength[:-1]
ks.free = False

kc = Parameter(**STANDARD_PARAMETERS.exp)
kc.set(min=-10, max=10)

fwhm = Parameter(**STANDARD_PARAMETERS.fwhm)
fwhm.value = 1
fwhm.set(min=0.1, max=32)

rin = Parameter(**STANDARD_PARAMETERS.rin)
rin.value = 1
rin.set(min=0.1, max=32)

flor = Parameter(**STANDARD_PARAMETERS.fr)
flor.value = 0.4
flor.free = True

a = Parameter(**STANDARD_PARAMETERS.a)
a.value = 0.5

phi = Parameter(**STANDARD_PARAMETERS.phi)
phi.value = 50

params = {"fs": fs, "fc": fc, "flor": flor, "fwhm": fwhm,
          "rin": rin, "kc": kc, "inc": inc, "pa": pa, "a": a, "phi": phi}
labels = [label for label in params]

OPTIONS.model.constant_params = {"wl0": 2.15, "ks": ks}
OPTIONS.model.components_and_params = [["StarHaloRing", params]]
OPTIONS.model.gridtype = "logarithmic"
OPTIONS.fit.method = "dynesty"

result_dir = Path("results/gravity")
model_name = "starHaloGaussLorRing"

plot.plot_overview(savefig=result_dir / f"{model_name}_data_overview.pdf")


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

    sampler = fitting.run_fit(**fit_params, ncores=ncores,
                              save_dir=result_dir, debug=False)

    theta, uncertainties = fitting.get_best_fit(
            sampler, **fit_params, method="quantile")

    components_and_params, shared_params = fitting.set_params_from_theta(theta)
    components = assemble_components(components_and_params, shared_params)
    rchi_sq = fitting.compute_observable_chi_sq(
            *fitting.compute_observables(components, wavelength), reduced=True)
    print(f"rchi_sq: {rchi_sq}")

    plot.plot_chains(sampler, labels, **fit_params,
                     savefig=result_dir / f"{model_name}_chains.pdf")
    plot.plot_corner(sampler, labels, **fit_params,
                     savefig=result_dir / f"{model_name}_corner.pdf")
    plot.plot_fit(components[0].inc(), components[0].pa(), components=components,
                  savefig=result_dir / f"{model_name}_fit_results.pdf")
