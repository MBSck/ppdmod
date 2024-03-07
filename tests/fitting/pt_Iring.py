from pathlib import Path

import astropy.units as u
import numpy as np
from ppdmod import data
from ppdmod import fitting
from ppdmod import plot
from ppdmod.basic_components import PointSource, Ring, assemble_components
from ppdmod.parameter import Parameter
from ppdmod.options import STANDARD_PARAMETERS, OPTIONS


OPTIONS.fit.data = ["vis"]
fits_file = Path("../data/aspro") / "model_pt_Iring.fits"
wavelength = [10] * u.um
data.set_fit_wavelengths(wavelength)
data.set_data([fits_file])
data.set_fit_weights()

vis = OPTIONS.data.vis
fr_ring, diameter = 0.6, 10
components = [PointSource(fr=fr_ring), Ring(fr=fr_ring, rin=diameter/2)]
vis_model = np.abs(np.sum([comp.compute_complex_vis(vis.ucoord, vis.vcoord, wavelength) for comp in components], axis=0))

assert np.allclose(vis.value, vis_model, rtol=1e-1)

chi_sq = fitting.compute_chi_sq(vis.value, vis.err, vis_model)
print(f"chi_sq: {chi_sq}")

fr = Parameter(**STANDARD_PARAMETERS.fr)
fr.value = 0.1
fr.free = True

OPTIONS.model.shared_params = {"fr": fr}
shared_params_labels = [f"sh_{label}" for label in OPTIONS.model.shared_params]

rin = Parameter(**STANDARD_PARAMETERS.rin)
rin.value = 0.8
rin.set(min=0., max=10)

ring = {"rin": rin}
ring_labels = [f"r_{label}" for label in ring]

OPTIONS.model.components_and_params = [["PointSource", {}], ["Ring", ring]]
OPTIONS.model.output = "non-physical"
OPTIONS.fit.method = "dynesty"

labels = ring_labels + shared_params_labels
result_dir = Path("results/Iring")
model_name = "pt_Iring"

if __name__ == "__main__":
    fit_params_emcee = {"nburnin": 1000, "nsteps": 2500, "nwalkers": 100}
    fit_params_dynesty = {"nlive": 1500, "sample": "rwalk", "bound": "multi"}

    if OPTIONS.fit.method == "emcee":
        fit_params = fit_params_emcee
        fit_params["discard"] = fit_params["nburnin"]
    else:
        fit_params = fit_params_dynesty

    sampler = fitting.run_fit(**fit_params, ncores=6, debug=False)
    theta, uncertainties = fitting.get_best_fit(sampler, **fit_params, method="quantile")

    components_and_params, shared_params = fitting.set_params_from_theta(theta)
    components = assemble_components(components_and_params, shared_params)

    plot.plot_chains(sampler, labels, **fit_params,
                     savefig=result_dir / f"{model_name}_chains.pdf")
    plot.plot_corner(sampler, labels, **fit_params,
                     savefig=result_dir / f"{model_name}_corner.pdf")
    plot.plot_fit(1, 0*u.deg, components=components,
                  savefig=result_dir / f"{model_name}_fit_results.pdf")

    assert np.isclose(theta[0], diameter/2, rtol=0.3)
    assert np.isclose(theta[1], fr_ring, rtol=0.1)
