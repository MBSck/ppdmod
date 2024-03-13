from pathlib import Path

import astropy.units as u

from ppdmod import data
from ppdmod import fitting
from ppdmod import plot
from ppdmod.basic_components import assemble_components
from ppdmod.parameter import Parameter
from ppdmod.options import STANDARD_PARAMETERS, OPTIONS


DATA_DIR = Path("../data/pionier/nChannels3")
OPTIONS.fit.data = ["vis2"]
OPTIONS.model.output = "non-physical"
wavelength = [1.7]*u.um
data.set_fit_wavelengths(wavelength)
fits_files = list((DATA_DIR).glob("*fits"))
data.set_data(fits_files)
data.set_fit_weights()

pa = Parameter(**STANDARD_PARAMETERS.pa)
pa.value = 162
pa.free = True

inc = Parameter(**STANDARD_PARAMETERS.inc)
inc.value = 0.5
inc.free = True

fr = Parameter(**STANDARD_PARAMETERS.fr)
fr.value = 0.4
fr.free = True

OPTIONS.model.shared_params = {"fr": fr, "pa": pa, "inc": inc}
shared_params_labels = [f"sh_{label}" for label in OPTIONS.model.shared_params]

fwhm = Parameter(**STANDARD_PARAMETERS.fwhm)
fwhm.value = 1
fwhm.set(min=0.1, max=32)

fr_lor = Parameter(**STANDARD_PARAMETERS.fr)
fr_lor.value = 0.4
fr_lor.free = True

gauss_lor = {"fr_lor": fr_lor, "fwhm": fwhm}
gauss_lor_labels = [f"gl_{label}" for label in gauss_lor]

OPTIONS.model.components_and_params = [
    ["PointSource", {}],
    ["GaussLorentzian", gauss_lor],
]

component_labels = ["Star", "Disk"]
OPTIONS.model.gridtype = "logarithmic"
OPTIONS.fit.method = "dynesty"

labels = gauss_lor_labels + shared_params_labels
result_dir = Path("results/pionier")
model_name = "lor_gauss"

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
                              save_dir=result_dir, debug=True)

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
    plot.plot_fit(theta[-2], theta[-1], components=components,
                  savefig=result_dir / f"{model_name}_fit_results.pdf")
