import os
import pickle
from datetime import datetime
from itertools import chain
from typing import List
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import astropy.units as u
import numpy as np

from ppdmod.analysis import save_fits
from ppdmod import basic_components
from ppdmod.fitting import run_fit, get_best_fit, compute_observables, \
    compute_observable_chi_sq, set_params_from_theta, ptform_sequential_radii, \
    ptform_one_disc
from ppdmod.data import set_data
from ppdmod.parameter import Parameter
from ppdmod.options import STANDARD_PARAMETERS, OPTIONS
from ppdmod.utils import load_data, qval_to_opacity, resample_and_convolve


def ptform(theta: List[float]) -> np.ndarray:
    return ptform_sequential_radii(theta, LABELS)


DATA_DIR = Path("../data")
wavelengths = {"hband": [1.7] * u.um, "kband": [2.15] * u.um,
               "lband": np.linspace(3.3, 3.8, 5) * u.um,
               "mband": np.linspace(4.6, 4.9, 3) * u.um,
               "nband": np.linspace(8, 13, 35) * u.um,
               }

fits_files = list((DATA_DIR / "fits" / "hd142527").glob("*fits"))
wavelength = np.concatenate((wavelengths["hband"], wavelengths["kband"],
                             wavelengths["lband"], wavelengths["mband"], wavelengths["nband"]))
data = set_data(fits_files, wavelengths=wavelength, fit_data=["flux", "vis"])

wl, value = load_data(DATA_DIR / "flux" / "hd142527" / "HD142527_stellar_model.txt", usecols=(0, 2))
star_flux = Parameter(**STANDARD_PARAMETERS.f)
star_flux.grid, star_flux.value = resample_and_convolve(OPTIONS.fit.wavelengths.value, wl, value)

wl, value = np.load(DATA_DIR / "opacities" / "hd142527_boekel_qval_silicates.npy")
kappa_abs = Parameter(**STANDARD_PARAMETERS.kappa_abs)
kappa_abs.value, kappa_abs.grid = resample_and_convolve(OPTIONS.fit.wavelengths.value, wl, value)

cont_opacity_file = DATA_DIR / "opacities" / "qval" / "Q_amorph_c_rv0.1.dat"
# cont_opacity_file = DATA_DIR / "opacities" / "qval" / "Q_iron_0.10um_dhs_0.70.dat"
wl, value = load_data(cont_opacity_file, load_func=qval_to_opacity)
kappa_cont = Parameter(**STANDARD_PARAMETERS.kappa_cont)
kappa_cont.value, kappa_cont.grid = resample_and_convolve(OPTIONS.fit.wavelengths.value, wl, value)

pa = Parameter(**STANDARD_PARAMETERS.pa)
inc = Parameter(**STANDARD_PARAMETERS.inc)

pa.value = 352
inc.value = 0.915
inc.free = True
pa.free = False

with open(DATA_DIR / "flux" / "hd142527" / "hd142527_dust_temperatures.pkl", "rb") as save_file:
    temps = pickle.load(save_file)

OPTIONS.model.constant_params = {
    "dim": 32, "dist": 158.51,
    "eff_temp": 6500, "eff_radius": 3.46,
    "temps": temps, "f": star_flux,
    "kappa_abs": kappa_abs,
    "kappa_cont": kappa_cont,
    "pa": pa, "cont_weight": 0.33}

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

rin.value = 1.
rout.value = 3.
sigma0.value = 1e-3
p.value = 0.5
c1.value = s1.value = 0.5
cont_weight.value = 0.40             # Relative contribution (adds to 1). Mass fractions
rin.unit = rout.unit = u.au
rout.free = True

rin.set(min=0, max=30)
rout.set(min=0, max=30)
rout.free = True
p.set(min=-10, max=10)
sigma0.set(min=0, max=1e-2)
cont_weight.set(min=0, max=1)

# one = {"rin": rin, "rout": rout, "p": p, "sigma0": sigma0, "cont_weight": cont_weight}
one = {"rin": rin, "rout": rout, "p": p, "sigma0": sigma0}

rin = Parameter(**STANDARD_PARAMETERS.rin)
rout = Parameter(**STANDARD_PARAMETERS.rout)
p = Parameter(**STANDARD_PARAMETERS.p)
sigma0 = Parameter(**STANDARD_PARAMETERS.sigma0)
c1 = Parameter(**STANDARD_PARAMETERS.c)
s1 = Parameter(**STANDARD_PARAMETERS.s)
cont_weight = Parameter(**STANDARD_PARAMETERS.cont_weight)

rin.value = 5
rout.value = 10
p.value = 0.5
sigma0.value = 1e-3
c1.value = s1.value = 0.5
cont_weight.value = 0.40             # Relative contribution (adds to 1). Mass fractions
rin.unit = rout.unit = u.au
rout.free = True

rin.set(min=0, max=30)
rout.set(min=0, max=30)
rout.free = True
p.set(min=-10, max=10)
sigma0.set(min=0, max=1)
cont_weight.set(min=0, max=1)

# two = {"rin": rin, "rout": rout, "p": p, "sigma0": sigma0, "cont_weight": cont_weight}
two = {"rin": rin, "rout": rout, "p": p, "sigma0": sigma0}

rin = Parameter(**STANDARD_PARAMETERS.rin)
rout = Parameter(**STANDARD_PARAMETERS.rout)
p = Parameter(**STANDARD_PARAMETERS.p)
sigma0 = Parameter(**STANDARD_PARAMETERS.sigma0)
c1 = Parameter(**STANDARD_PARAMETERS.c)
s1 = Parameter(**STANDARD_PARAMETERS.s)
cont_weight = Parameter(**STANDARD_PARAMETERS.cont_weight)

rin.value = 12
p.value = 0.5
sigma0.value = 1e-3
c1.value = s1.value = 0.5
cont_weight.value = 0.40             # Relative contribution (adds to 1). Mass fractions
rin.unit = rout.unit = u.au
rout.free = True

rin.set(min=0, max=30)
rout.set(min=0, max=30)
rout.free = True
p.set(min=-10, max=10)
sigma0.set(min=0, max=1e-1)
cont_weight.set(min=0, max=1)

# three = {"rin": rin, "p": p, "sigma0": sigma0, "cont_weight": cont_weight}
three = {"rin": rin, "p": p, "sigma0": sigma0, "cont_weight": cont_weight}

OPTIONS.model.shared_params = {"inc": inc}
shared_param_labels = [f"{label}-sh" for label in OPTIONS.model.shared_params]
shared_param_units = [value.unit for value in OPTIONS.model.shared_params.values()]

OPTIONS.model.components_and_params = [
    ["Star", star],
    ["GreyBody", one],
    ["GreyBody", two],
    # ["GreyBody", three],
]

ring_labels = [[f"{key}-{index}" for key in ring]
    for index, ring in enumerate([one, two, three], start=1)]
ring_units = [[value.unit for value in ring.values()] for ring in [one, two, three]]

LABELS = list(chain.from_iterable([star_labels, *ring_labels][:len(OPTIONS.model.components_and_params)]))
LABELS += shared_param_labels
UNITS = list(chain.from_iterable([star_units, *ring_units][:len(OPTIONS.model.components_and_params)]))
UNITS += shared_param_units

component_labels = ["Star", "Inner Ring", "Outer Ring", "Last Ring"]
component_labels = component_labels[:len(OPTIONS.model.components_and_params)]
OPTIONS.fit.method = "dynesty"

model_result_dir = Path("../model_results/")
day_dir = model_result_dir / str(datetime.now().date())
dir_name = f"results_model_{datetime.now().strftime('%H:%M:%S')}"
result_dir = day_dir / dir_name
result_dir.mkdir(parents=True, exist_ok=True)
np.save(result_dir / "labels.npy", LABELS)
np.save(result_dir / "units.npy", UNITS)

components = basic_components.assemble_components(
        OPTIONS.model.components_and_params,
        OPTIONS.model.shared_params)
rchi_sq = compute_observable_chi_sq(
        *compute_observables(components), reduced=True)
print(f"rchi_sq: {rchi_sq:.2f}")


if __name__ == "__main__":
    ncores = 70
    fit_params = {"nlive_init": 2000, "ptform": ptform}
    sampler = run_fit(**fit_params, ncores=ncores,
                      method="dynamic", save_dir=result_dir,
                      debug=False)

    theta, uncertainties = get_best_fit(sampler, **fit_params)
    components_and_params, shared_params = set_params_from_theta(theta)
    components = basic_components.assemble_components(
            components_and_params, shared_params)
    rchi_sq = compute_observable_chi_sq(
            *compute_observables(components), reduced=True)
    print(f"rchi_sq: {rchi_sq:.2f}")

    save_fits(
        components, component_labels,
        fit_hyperparameters=fit_params, ncores=ncores,
        save_dir=result_dir / "model.fits",
        object_name="HD142527")
