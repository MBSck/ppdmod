import os
from datetime import datetime
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import astropy.units as u
import numpy as np

from ppdmod.analysis import save_fits
from ppdmod.basic_components import assemble_components
from ppdmod.data import set_data
from ppdmod.fitting import (
    compute_sed_chi_sq,
    get_best_fit,
    lnprob_sed,
    ptform_sed,
    run_fit,
    set_params_from_theta,
)
from ppdmod.options import OPTIONS, STANDARD_PARAMETERS
from ppdmod.parameter import Parameter
from ppdmod.plot import plot_sed


def ptform(theta):
    return ptform_sed(theta, LABELS)


DATA_DIR = Path("../data")
# fits_dir = DATA_DIR / "fits" / "hd142527" / "sed_fit" / "averaged"
fits_dir = DATA_DIR / "fits" / "hd142527" / "sed_fit" / "downsampled"
# fits_dir = DATA_DIR / "fits" / "hd142527" / "sed_fit" / "only_high"
# fits_dir = DATA_DIR / "fits" / "hd142527" / "sed_fit" / "only_low"

# wavelength_range = None
wavelength_range = [8., 13.1] * u.um
data = set_data(list(fits_dir.glob("*fits")), wavelengths="all",
                wavelength_range=wavelength_range, fit_data=["flux"])

WAVELENGTHS = OPTIONS.fit.wavelengths
OPACITY_DIR = DATA_DIR / "opacities"
GRF_DIR = OPACITY_DIR / "grf"

SHORTNAMES = ["pyrox", "enst", "forst", "sil", "oliv"]
NAMES = ["MgPyroxene", "Enstatite", "Forsterite", "Silica", "Olivine"]
NAMES = dict(zip(SHORTNAMES, NAMES))

OPTIONS.model.constant_params = {}
for shortname, name in NAMES.items():
    for size, value in {"small": 0.1, "large": 2.0}.items():
        grid, value = np.loadtxt(GRF_DIR / f"{name}{value}.Combined.Kappa", usecols=(0, 2), unpack=True)
        param_name, param = f"kappa_{shortname}_{size}", Parameter(**STANDARD_PARAMETERS.kappa_abs)
        param.grid, param.value = grid, value
        param.shortname = param.name = param_name
        OPTIONS.model.constant_params[param_name] = param

kappa_cont = Parameter(**STANDARD_PARAMETERS.kappa_cont)
grid, value = np.load(OPACITY_DIR / "optool" / "preibisch_amorph_c_rv0.1.npy")
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

cont_weight = Parameter(**STANDARD_PARAMETERS.cont_weight)
cont_weight.set(min=0, max=100)
cont_weight.unit = u.pct
cont_weight.value = 54

pah_weight = Parameter(**STANDARD_PARAMETERS.cont_weight)
pah_weight.shortname = pah_weight.name = "pah_weight"
pah_weight.description = "The mass fraction for the PAHs"
pah_weight.set(min=0, max=20)
pah_weight.unit, pah_weight.value = u.one, 1.66

fr = Parameter(**STANDARD_PARAMETERS.fr)
fr.description = "Opacity scaling term"
fr.set(min=15, max=25)
fr.free, fr.value = True, 17.46

sed = {"tempc": tempc, "pah_weight": pah_weight, "cont_weight": cont_weight, "fr": fr}
weights = [[11.23, 13.40], [5.67, 5.85], [4.09, 3.77], [0.6, 0.24], [0.10, 0.10]]
for w, key in zip(weights, NAMES.keys()):
    for index, size in enumerate(["small", "large"]):
        weight_name = f"{key}_{size}_weight"
        weight = Parameter(**STANDARD_PARAMETERS.cont_weight)
        weight.shortname = weight.name = weight_name
        weight.description = f"The mass fraction for {size} {key}"
        weight.set(min=0, max=100)
        weight.value, weight.unit = w[index], u.pct
        sed[weight_name] = weight

OPTIONS.model.components_and_params = [["SED", sed]]
LABELS, UNITS = [key for key in sed], [value.unit for value in sed.values()]
component_labels = ["SED"]
OPTIONS.fit.method = "dynesty"

result_dir = Path("../model_results/") / "sed_fit"
day_dir = result_dir / str(datetime.now().date())
dir_name = f"results_model_{datetime.now().strftime('%H:%M:%S')}"
result_dir = day_dir / dir_name
result_dir.mkdir(parents=True, exist_ok=True)

np.save(result_dir / "labels.npy", LABELS)
np.save(result_dir / "units.npy", UNITS)

components = assemble_components(
        OPTIONS.model.components_and_params,
        OPTIONS.model.shared_params)

rchi_sq = compute_sed_chi_sq(
    components[0].compute_flux(OPTIONS.fit.wavelengths), reduced=True)
print(f"rchi_sq: {rchi_sq:.2f}")


if __name__ == "__main__":
    ncores = 70
    fit_params = {"nlive_init": 2000, "lnprob": lnprob_sed, "ptform": ptform}
    sampler = run_fit(**fit_params, ncores=ncores,
                      method="dynamic", save_dir=result_dir,
                      debug=False)

    theta, uncertainties = get_best_fit(sampler, **fit_params)
    components_and_params, shared_params = set_params_from_theta(theta)
    components = assemble_components(
            components_and_params, shared_params)

    rchi_sq = compute_sed_chi_sq(
        components[0].compute_flux(OPTIONS.fit.wavelengths), reduced=True)
    print(f"rchi_sq: {rchi_sq:.2f}")

    plot_sed([7.5, 14] * u.um, components, scaling="nu", save_dir=result_dir)
    plot_sed([7.5, 14] * u.um, components, scaling=None, save_dir=result_dir)

    save_fits(
        components, component_labels,
        fit_hyperparameters=fit_params, ncores=ncores,
        save_dir=result_dir / "sed.fits",
        object_name="HD142527")
