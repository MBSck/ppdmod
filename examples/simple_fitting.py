from datetime import datetime
from pathlib import Path

import astropy.units as u
import numpy as np

# from ppdmod import analysis
from ppdmod import basic_components
from ppdmod import data
from ppdmod import fitting
from ppdmod import plot
from ppdmod.parameter import Parameter
from ppdmod.options import STANDARD_PARAMETERS, OPTIONS


DATA_DIR = Path("../tests/data")
OPTIONS.fit.data = ["vis2", "t3"]
OPTIONS.model.output = "non-physical"
wavelengths = [1.6]*u.um
data.set_fit_wavelengths(wavelengths)
fits_files = list((DATA_DIR / "fits").glob("*PION*fits"))
data.set_data(fits_files)
data.set_fit_weights()

wavelength_axes = list(
    map(lambda x: x.wavelength, OPTIONS.data.readouts))
wavelength_axes = np.sort(np.unique(np.concatenate(wavelength_axes)))

pa = Parameter(**STANDARD_PARAMETERS.pa)
pa.value = 162

inc = Parameter(**STANDARD_PARAMETERS.inc)
inc.value = 0.5

fr = Parameter(**STANDARD_PARAMETERS.fr)
fr.value = 0.4
fr.free = True

OPTIONS.model.shared_params = {"fr": fr, "pa": pa, "inc": inc}
shared_params_labels = [f"sh_{label}" for label in OPTIONS.model.shared_params]

fwhm = Parameter(**STANDARD_PARAMETERS.fwhm)
fwhm.value = 1
fwhm.set(min=0.1, max=3)

fr_lor = Parameter(**STANDARD_PARAMETERS.fr)
fr_lor.value = 0.4
fr_lor.free = True

gauss_lor = {"fr_lor": fr_lor, "fwhm": fwhm}
gauss_lor_labels = [f"gl_{label}" for label in gauss_lor]

OPTIONS.model.components_and_params = [
    ["PointSource", {}],
    ["GaussLorentzian", gauss_lor],
]

labels = gauss_lor_labels + shared_params_labels

component_labels = ["Star", "Disk"]
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
plot.plot_observables("hd142666", [3, 12]*u.um, components,
                      save_dir=pre_fit_dir)

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
            save_dir=post_fit_dir, debug=False)

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

    plot.plot_observables("hd142666", [3, 12]*u.um, components,
                          save_dir=post_fit_dir)

    # analysis.save_fits(
    #         4096, 0.1, distance,
    #         components, component_labels,
    #         opacities=[kappa_abs, kappa_cont],
    #         savefits=post_fit_dir / "model.fits",
    #         object_name="HD 142666", **fit_params, ncores=ncores)

    axis_ratio = shared_params["elong"]
    compression = shared_params["pa"]

    plot.plot_fit(
            axis_ratio, compression,
            components=components,
            savefig=post_fit_dir / "fit_results.pdf")
