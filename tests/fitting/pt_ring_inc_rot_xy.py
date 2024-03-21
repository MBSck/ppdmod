from pathlib import Path

import astropy.units as u
import numpy as np
from ppdmod import data
from ppdmod import fitting
from ppdmod import plot
from ppdmod.basic_components import PointSource, Ring, assemble_components
from ppdmod.parameter import Parameter
from ppdmod.options import STANDARD_PARAMETERS, OPTIONS
from ppdmod.utils import compute_vis, compute_t3


# TODO: Change the errors and do this test again.
OPTIONS.model.output = "non-physical"
fits_file = Path("../data/aspro") / "model_pt_ring_inc_rot_xy.fits"
wavelength = [10] * u.um
data.set_data([fits_file], wavelengths=wavelength, fit_data=["vis", "t3"])

vis, t3 = OPTIONS.data.vis, OPTIONS.data.t3
fr_point, fr_ring, diameter, inclination, pos_angle, width, xcoord, ycoord = 0.4, 0.6, 10, 0.351, 33, 2, 4, 7
components = [PointSource(fr=fr_point, x=xcoord, y=ycoord),
              Ring(fr=fr_ring, rin=diameter/2, width=width, inc=inclination, pa=pos_angle, thin=False)]
vis_model = compute_vis(np.sum([comp.compute_complex_vis(vis.ucoord, vis.vcoord, wavelength) for comp in components], axis=0))
t3_model = compute_t3(np.sum([comp.compute_complex_vis(t3.u123coord, t3.v123coord, wavelength) for comp in components], axis=0))

assert np.allclose(vis.value, vis_model, rtol=1e-1)
assert np.allclose(t3.value, t3_model, rtol=1e0)

chi_sq = fitting.compute_chi_sq(vis.value, vis.err, vis_model) \
    + fitting.compute_chi_sq(t3.value, t3.err, t3_model)
rchi_sq = fitting.compute_observable_chi_sq(
        *fitting.compute_observables(components, wavelength), reduced=True)
print(f"chi_sq: {chi_sq}", f"rchi_sq: {rchi_sq}")

fr_pt = Parameter(**STANDARD_PARAMETERS.fr)
fr_pt.value = 0.1
fr_pt.free = True

point = {"fr": fr_pt}
point_labels = [f"pt_{label}" for label in point]

fr_r = Parameter(**STANDARD_PARAMETERS.fr)
fr_r.value = 0.1
fr_r.free = True

inc = Parameter(**STANDARD_PARAMETERS.inc)
inc.value = 0
inc.free = True

pa = Parameter(**STANDARD_PARAMETERS.pa)
pa.value = 0
pa.free = True

x = Parameter(**STANDARD_PARAMETERS.x)
x.set(min=-10, max=10)
x.free = True

y = Parameter(**STANDARD_PARAMETERS.y)
y.set(min=-10, max=10)
y.free = True

point_source = {"x": x, "y": y, "fr": fr_pt}
point_source_labels = [f"pt_{label}" for label in point_source]

rin = Parameter(**STANDARD_PARAMETERS.rin)
rin.value = 0.8
rin.set(min=0., max=10)

w = Parameter(**STANDARD_PARAMETERS.width)
w.value = 4
w.set(min=0, max=5)

ring = {"fr": fr_r, "rin": rin, "width": w, "inc": inc, "pa": pa}
ring_labels = [f"r_{label}" for label in ring]

OPTIONS.model.components_and_params = [["PointSource", point_source], ["Ring", ring]]
OPTIONS.fit.method = "dynesty"

labels = point_source_labels + ring_labels
result_dir = Path("results/ring")
model_name = "pt_ring_inc_rot_xy"


if __name__ == "__main__":
    fit_params_emcee = {"nburnin": 1000, "nsteps": 2500, "nwalkers": 100}
    fit_params_dynesty = {"nlive": 1500, "sample": "rwalk", "bound": "multi"}

    if OPTIONS.fit.method == "emcee":
        fit_params = fit_params_emcee
        fit_params["discard"] = fit_params["nburnin"]
    else:
        fit_params = fit_params_dynesty

    sampler = fitting.run_fit(**fit_params, ncores=6, debug=True)
    theta, uncertainties = fitting.get_best_fit(sampler, **fit_params, method="quantile")

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

    assert np.isclose(theta[0], xcoord, rtol=0.3)
    assert np.isclose(theta[1], ycoord, rtol=0.3)
    assert np.isclose(theta[2], fr_point, rtol=0.3)
    assert np.isclose(theta[3], fr_ring, rtol=0.3)
    assert np.isclose(theta[4], diameter/2, rtol=0.3)
    assert np.isclose(theta[5], width, rtol=0.3)
    assert np.isclose(theta[-2], inclination, rtol=0.1)
    assert np.isclose(theta[-1], pos_angle, rtol=0.1)
