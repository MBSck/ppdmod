import os
import pickle
from datetime import datetime
from itertools import chain
from pathlib import Path
from typing import List

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import astropy.units as u
import numpy as np

from ppdmod import basic_components
from ppdmod.data import set_data
from ppdmod.fitting import (
    compute_observable_chi_sq,
    compute_observables,
    get_best_fit,
    ptform_one_disc,
    ptform_sequential_radii,
    run_fit,
    set_params_from_theta,
)
from ppdmod.options import OPTIONS, STANDARD_PARAMETERS
from ppdmod.parameter import Parameter
from ppdmod.utils import load_data, qval_to_opacity


def ptform(theta: List[float]) -> np.ndarray:
    return ptform_sequential_radii(theta, LABELS)


DATA_DIR = Path(__file__).parent.parent / "data"
wavelengths = {
    "hband": [1.7] * u.um,
    "kband": [2.15] * u.um,
    "lband": np.linspace(3.1, 3.4, 5) * u.um,
    "mband": np.linspace(4.7, 4.9, 3) * u.um,
    "nband": np.linspace(8, 15, 35) * u.um,
}

fits_files = list((DATA_DIR / "fits" / "hd142527").glob("*fits"))
wavelength = np.concatenate(
    (
        wavelengths["hband"],
        wavelengths["kband"],
        wavelengths["lband"],
        wavelengths["mband"],
        wavelengths["nband"],
    )
)
data = set_data(
    fits_files,
    wavelengths=wavelength,
    fit_data=["flux", "vis"],
    set_std_err=["mband"],
    # weights = [13.77604942, 1]
)
WAVELENGTHS = OPTIONS.fit.wavelengths

grid, value = np.load(DATA_DIR / "flux" / "hd142527" / "HD142527_stellar_model.npy")
star_flux = Parameter(**STANDARD_PARAMETERS.f)
star_flux.grid, star_flux.value = grid, value

SOURCE_DIR = DATA_DIR / "model_results" / "hd142527"

method = "grf"
grid, value = np.load(SOURCE_DIR / f"silicate_{method}_opacities.npy")
kappa_abs = Parameter(**STANDARD_PARAMETERS.kappa_abs)
kappa_abs.grid, kappa_abs.value = grid, value

grid, value = load_data(
    DATA_DIR / "opacities" / "qval" / "Q_amorph_c_rv0.1.dat", load_func=qval_to_opacity
)
kappa_cont = Parameter(**STANDARD_PARAMETERS.kappa_cont)
kappa_cont.grid, kappa_cont.value = grid, value

pa = Parameter(**STANDARD_PARAMETERS.pa)
inc = Parameter(**STANDARD_PARAMETERS.inc)

pa.value = 352
inc.value = 0.915
inc.free = True
pa.free = False

OPTIONS.model.constant_params = {
    "dim": 32,
    "dist": 158.51,
    "eff_temp": 6500,
    "eff_radius": 3.46,
    "f": star_flux,
    "kappa_abs": kappa_abs,
    "kappa_cont": kappa_cont,
    "pa": pa,
}


# with open(
#     DATA_DIR / "flux" / "hd142527" / "hd142527_dust_temperatures.pkl", "rb"
# ) as save_file:
#     temps = pickle.load(save_file)
#     OPTIONS.model.constant_params["temps"] = temps

x = Parameter(**STANDARD_PARAMETERS.x)
y = Parameter(**STANDARD_PARAMETERS.y)
x.free = y.free = True
star = {}
star_labels = [rf"{label}-\star" for label in star]
star_units = [value.unit for value in star.values()]

rin = Parameter(**STANDARD_PARAMETERS.rin)
rout = Parameter(**STANDARD_PARAMETERS.rout)
p = Parameter(**STANDARD_PARAMETERS.p)
sigma0 = Parameter(**STANDARD_PARAMETERS.sigma0)
c1 = Parameter(**STANDARD_PARAMETERS.c)
s1 = Parameter(**STANDARD_PARAMETERS.s)
cont_weight = Parameter(**STANDARD_PARAMETERS.cont_weight)

rin.set(min=0, max=30)
rout.set(min=0, max=30)
rout.free = True
p.set(min=-20, max=20)
sigma0.set(min=0, max=1e-1)

rin.value = 0.5
rout.value = 1.5
p.value = 0.5
sigma0.value = 1e-3

one = {"rin": rin, "rout": rout, "p": p, "sigma0": sigma0, "cont_weight": cont_weight}
# one = {"rin": rin, "rout": rout, "p": p, "sigma0": sigma0}

rin = Parameter(**STANDARD_PARAMETERS.rin)
rout = Parameter(**STANDARD_PARAMETERS.rout)
p = Parameter(**STANDARD_PARAMETERS.p)
sigma0 = Parameter(**STANDARD_PARAMETERS.sigma0)
c1 = Parameter(**STANDARD_PARAMETERS.c)
s1 = Parameter(**STANDARD_PARAMETERS.s)
cont_weight = Parameter(**STANDARD_PARAMETERS.cont_weight)

rin.set(min=0, max=50)
rout.set(min=0, max=50)
rout.free = False
p.set(min=-30, max=20)
sigma0.set(min=0, max=1e-1)

rin.value = 2
rout.value = 4
p.value = 0.5
sigma0.value = 1e-3

# two = {"rin": rin, "rout": rout, "p": p, "sigma0": sigma0, "cont_weight": cont_weight}
two = {"rin": rin, "p": p, "sigma0": sigma0, "cont_weight": cont_weight}

OPTIONS.model.shared_params = {"inc": inc}
shared_param_labels = [f"{label}-sh" for label in OPTIONS.model.shared_params]
shared_param_units = [value.unit for value in OPTIONS.model.shared_params.values()]

OPTIONS.model.components_and_params = [
    ["Star", star],
    ["GreyBody", one],
    ["GreyBody", two],
]

ring_labels = [
    [f"{key}-{index}" for key in ring] for index, ring in enumerate([one, two], start=1)
]
ring_units = [[value.unit for value in ring.values()] for ring in [one, two]]

LABELS = list(
    chain.from_iterable(
        [star_labels, *ring_labels][: len(OPTIONS.model.components_and_params)]
    )
)
LABELS += shared_param_labels
UNITS = list(
    chain.from_iterable(
        [star_units, *ring_units][: len(OPTIONS.model.components_and_params)]
    )
)
UNITS += shared_param_units

component_labels = ["Star", "Inner Ring", "Outer Ring"]
OPTIONS.fit.method = "dynesty"

result_dir = Path("../model_results/") / "disc_fit"
day_dir = result_dir / str(datetime.now().date())
dir_name = f"results_model_{datetime.now().strftime('%H:%M:%S')}"
result_dir = day_dir / dir_name
result_dir.mkdir(parents=True, exist_ok=True)
np.save(result_dir / "labels.npy", LABELS)
np.save(result_dir / "units.npy", UNITS)

components = basic_components.assemble_components(
    OPTIONS.model.components_and_params, OPTIONS.model.shared_params
)
# rchi_sqs = compute_observable_chi_sq(
#     *compute_observables(components),
#     ndim=len(UNITS),
#     method="linear",
#     reduced=True,
# )
# print(f"rchi_sq: {rchi_sqs[0]:.2f}")


if __name__ == "__main__":
    ncores = 70
    fit_params = {"nlive_init": 2000, "ptform": ptform}
    sampler = run_fit(
        **fit_params, ncores=ncores, method="dynamic", save_dir=result_dir, debug=False
    )

    theta, uncertainties = get_best_fit(sampler, **fit_params)
    components_and_params, shared_params = set_params_from_theta(theta)
    components = basic_components.assemble_components(
        components_and_params, shared_params
    )

    np.save(result_dir / "theta.npy", theta)
    np.save(result_dir / "uncertainties.npy", uncertainties)
    with open(result_dir / "components.pkl", "wb") as file:
        pickle.dump(components, file)

    rchi_sqs = compute_observable_chi_sq(
        *compute_observables(components),
        ndim=theta.size,
        method="linear",
        reduced=True,
    )
    print(f"rchi_sq: {rchi_sqs[0]:.2f}")
