import os
import pickle
from datetime import datetime
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import astropy.units as u
import numpy as np

from ppdmod.basic_components import assemble_components
from ppdmod.data import set_data
from ppdmod.fitting import (
    compute_sed_chi_sq,
    get_best_fit,
    lnprob_nband_fit,
    ptform_lband_fit,
    run_fit,
    set_params_from_theta,
)
from ppdmod.options import OPTIONS, STANDARD_PARAMETERS
from ppdmod.parameter import Parameter
from ppdmod.utils import load_data, qval_to_opacity


def ptform(theta):
    return ptform_lband_fit(theta, LABELS)


DATA_DIR = Path(__file__).parent.parent / "data"
fits_dir = DATA_DIR / "fits" / "hd142527" / "nband_fit" / "averaged"

wavelength_range = [8.0, 15] * u.um
data = set_data(
    list(fits_dir.glob("*fits")),
    wavelengths="all",
    wavelength_range=wavelength_range,
    fit_data=["flux"],
)

WAVELENGTHS = OPTIONS.fit.wavelengths
OPACITY_DIR = DATA_DIR / "opacities"
GRF_DIR = OPACITY_DIR / "grf"

SHORTNAMES = ["pyrox", "enst", "forst", "sil", "oliv"]
NAMES = ["MgPyroxene", "Enstatite", "Forsterite", "Silica", "Olivine"]
NAMES = dict(zip(SHORTNAMES, NAMES))

OPTIONS.model.constant_params = {}
for shortname, name in NAMES.items():
    for size, value in {"small": 0.1, "large": 2.0}.items():
        grid, value = np.loadtxt(
            GRF_DIR / f"{name}{value}.Combined.Kappa", usecols=(0, 2), unpack=True
        )
        param_name, param = (
            f"kappa_{shortname}_{size}",
            Parameter(**STANDARD_PARAMETERS.kappa_abs),
        )
        param.grid, param.value = grid, value
        param.shortname = param.name = param_name
        OPTIONS.model.constant_params[param_name] = param

kappa_cont = Parameter(**STANDARD_PARAMETERS.kappa_cont)
grid, value = load_data(
    DATA_DIR / "opacities" / "qval" / "Q_amorph_c_rv0.1.dat", load_func=qval_to_opacity
)
kappa_cont.grid, kappa_cont.value = grid, value
OPTIONS.model.constant_params["kappa_cont"] = kappa_cont

pah = Parameter(**STANDARD_PARAMETERS.pah)
grid, value = np.loadtxt(OPACITY_DIR / "boekel" / "PAH.kappa", unpack=True)
pah.grid, pah.value = grid, value
OPTIONS.model.constant_params["pah"] = pah

tempc = Parameter(**STANDARD_PARAMETERS.temp0)
tempc.shortname = tempc.name = "tempc"
tempc.description = "The temperature of the black body"
tempc.value = 390.08

weight_cont = Parameter(**STANDARD_PARAMETERS.cont_weight)
weight_cont.set(min=0, max=100)
weight_cont.unit = u.pct
weight_cont.value = 54

scale_pah = Parameter(**STANDARD_PARAMETERS.cont_weight)
scale_pah.shortname = scale_pah.name = "scale_pah"
scale_pah.description = "The mass fraction for the PAHs"
scale_pah.set(min=0, max=20)
scale_pah.unit, scale_pah.value = u.one, 1.66

f = Parameter(**STANDARD_PARAMETERS.fr)
f.description = "Offset term"
f.name = f.shortname = "f"
f.unit = u.one
f.set(min=15, max=25)
f.free, f.value = True, 17.46

nband_fit = {"tempc": tempc, "scale_pah": scale_pah, "weight_cont": weight_cont, "f": f}
weights = [[11.23, 13.40], [5.67, 5.85], [4.09, 3.77], [0.6, 0.24], [0.10, 0.10]]
for w, key in zip(weights, NAMES.keys()):
    for index, size in enumerate(["small", "large"]):
        weight_name = f"weight_{key}_{size}"
        weight = Parameter(**STANDARD_PARAMETERS.cont_weight)
        weight.shortname = weight.name = weight_name
        weight.description = f"The mass fraction for {size} {key}"
        weight.set(min=0, max=100)
        weight.value, weight.unit = w[index], u.pct
        nband_fit[weight_name] = weight

OPTIONS.model.components_and_params = [["NBandFit", nband_fit]]
LABELS, UNITS = [key for key in nband_fit], [value.unit for value in nband_fit.values()]
component_labels = ["NBandFit"]

result_dir = Path("../model_results/") / "nband_fit"
day_dir = result_dir / str(datetime.now().date())
dir_name = f"results_model_{datetime.now().strftime('%H:%M:%S')}"
result_dir = day_dir / dir_name
result_dir.mkdir(parents=True, exist_ok=True)

np.save(result_dir / "labels.npy", LABELS)
np.save(result_dir / "units.npy", UNITS)

components = assemble_components(
    OPTIONS.model.components_and_params, OPTIONS.model.shared_params
)

rchi_sq = compute_sed_chi_sq(
    components[0].compute_flux(OPTIONS.fit.wavelengths),
    ndim=len(UNITS),
    method="linear",
)
print(f"rchi_sq: {rchi_sq:.2f}")


if __name__ == "__main__":
    ncores = 70
    fit_params = {"nlive_init": 2000, "lnprob": lnprob_nband_fit, "ptform": ptform}
    sampler = run_fit(
        **fit_params, ncores=ncores, method="dynamic", save_dir=result_dir, debug=False
    )

    theta, uncertainties = get_best_fit(sampler, **fit_params)
    components_and_params, shared_params = set_params_from_theta(theta)
    components = assemble_components(components_and_params, shared_params)

    with open(result_dir / "components.pkl", "wb") as file:
        pickle.dump(components, file)

    rchi_sq = compute_sed_chi_sq(
        components[0].compute_flux(OPTIONS.fit.wavelengths),
        ndim=len(UNITS),
        method="linear",
    )
    print(f"rchi_sq: {rchi_sq:.2f}")
