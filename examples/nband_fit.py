import os
import pickle
from datetime import datetime
from pathlib import Path

from ppdmod.basic_components import NBandFit

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import astropy.units as u
import numpy as np

from ppdmod.data import set_data
from ppdmod.fitting import (
    compute_nband_fit_chi_sq,
    get_best_fit,
    lnprob_nband_fit,
    ptform_nband_fit,
    run_fit,
    set_components_from_theta,
    get_labels,
)
from ppdmod.options import OPTIONS
from ppdmod.parameter import Parameter
from ppdmod.utils import load_data, qval_to_opacity


def ptform(theta):
    return ptform_nband_fit(theta, LABELS)


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

constant_params = {}
for shortname, name in NAMES.items():
    for size, value in {"small": 0.1, "large": 2.0}.items():
        grid, value = np.loadtxt(
            GRF_DIR / f"{name}{value}.Combined.Kappa", usecols=(0, 2), unpack=True
        )
        param_name = f"kappa_{shortname}_{size}"
        param = Parameter(
            grid=grid,
            value=value,
            shortname=param_name,
            name=param_name,
            base="kappa_abs",
        )
        constant_params[param_name] = param

grid, value = load_data(
    DATA_DIR / "opacities" / "qval" / "Q_amorph_c_rv0.1.dat", load_func=qval_to_opacity
)
kappa_cont = Parameter(grid=grid, value=value, base="kappa_cont")
grid, value = np.loadtxt(OPACITY_DIR / "boekel" / "PAH.kappa", unpack=True)
pah = Parameter(grid=grid, value=value, base="pah")
scale_pah = Parameter(
    value=1.66,
    unit=u.one,
    min=0,
    max=20,
    shortname="scale_pah",
    name="scale_pah",
    base="weight_cont",
)
f = Parameter(
    value=17.46, min=15, max=25, free=True, unit=u.one, description="Offset", base="f"
)

weight_params = {}
weights = [[11.23, 13.40], [5.67, 5.85], [4.09, 3.77], [0.6, 0.24], [0.10, 0.10]]
for w, key in zip(weights, NAMES.keys()):
    for index, size in enumerate(["small", "large"]):
        weight_name = f"weight_{key}_{size}"
        weight = Parameter(
            value=w[index],
            shortname=weight_name,
            name=weight_name,
            description=f"The mass fraction for {size} {key}",
            base="weight_cont",
        )
        weight_params[weight_name] = weight

nband_fit = NBandFit(
    tempc=390.08,
    scale_pah=scale_pah,
    weight_cont=54,
    f=f,
    kappa_cont=kappa_cont,
    pah=pah,
    **weight_params,
    **constant_params,
)
OPTIONS.model.components = components = [nband_fit]
LABELS = get_labels(components)

result_dir = Path("../model_results/") / "nband_fit"
day_dir = result_dir / str(datetime.now().date())
dir_name = f"results_model_{datetime.now().strftime('%H:%M:%S')}"
result_dir = day_dir / dir_name
result_dir.mkdir(parents=True, exist_ok=True)

rchi_sq = compute_nband_fit_chi_sq(
    components[0].compute_flux(OPTIONS.fit.wavelengths),
    ndim=len(LABELS),
    method="linear",
    reduced=True,
)
print(f"rchi_sq: {rchi_sq:.2f}")


if __name__ == "__main__":
    ncores = 50
    fit_params = {"nlive_init": 2000, "lnprob": lnprob_nband_fit, "ptform": ptform}
    sampler = run_fit(**fit_params, ncores=ncores, save_dir=result_dir, debug=False)

    theta, uncertainties = get_best_fit(sampler, **fit_params)
    components = set_components_from_theta(theta)

    with open(result_dir / "components.pkl", "wb") as file:
        pickle.dump(components, file)

    rchi_sq = compute_nband_fit_chi_sq(
        components[0].compute_flux(OPTIONS.fit.wavelengths),
        ndim=theta.size,
        method="linear",
        reduced=True,
    )
    print(f"rchi_sq: {rchi_sq:.2f}")
