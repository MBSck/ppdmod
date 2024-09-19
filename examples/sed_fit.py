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
from ppdmod import basic_components
from ppdmod.fitting import run_fit, compute_sed, get_best_fit, \
    compute_chi_sq, set_params_from_theta, get_priors, lnprob_sed, ptform_sed
from ppdmod.data import set_data, get_all_wavelengths
from ppdmod.parameter import Parameter
from ppdmod.plot import plot_corner, plot_sed
from ppdmod.options import STANDARD_PARAMETERS, OPTIONS


def ptform(theta):
    return ptform_sed(theta, LABELS)


DATA_DIR = Path("../tests/data")

OPTIONS.model.output = "non-normed"
fits_file = [DATA_DIR / "fits" / "hd142527" / "sed_fit" / "hd142527_average_sed.fits"]
data = set_data(fits_file, wavelengths="all", fit_data=["flux"])
all_wavelengths = get_all_wavelengths()

OPACITY_DIR = DATA_DIR / "opacities"
GRF_DIR = OPACITY_DIR / "grf"

# TODO: Order of these might be important for the fit -> Check that
SHORTNAMES = ["pyrox", "enst", "forst", "sil", "oliv"]
NAMES = ["MgPyroxene", "Enstatite", "Forsterite", "Silica", "Olivine"]
NAMES = dict(zip(SHORTNAMES, NAMES))

OPTIONS.model.constant_params = {}
for shortname, name in NAMES.items():
    for size, value in {"small": 0.1, "large": 2.0}.items():
        wl, kappa = np.loadtxt(GRF_DIR / f"{name}{value}.Combined.Kappa", usecols=(0, 2), unpack=True)
        param_name, param = f"kappa_{shortname}_{size}", Parameter(**STANDARD_PARAMETERS.kappa_abs)
        param.grid, param.value = wl, kappa
        param.shortname = param.name = param_name
        OPTIONS.model.constant_params[param_name] = param

kappa_cont = Parameter(**STANDARD_PARAMETERS.kappa_cont)
kappa_cont.grid, kappa_cont.value = np.load(OPACITY_DIR / "optool" / "preibisch_amorph_c_rv0.1.npy")
OPTIONS.model.constant_params["kappa_cont"] = kappa_cont

pah = Parameter(**STANDARD_PARAMETERS.pah)
pah.grid, pah.value = np.loadtxt(OPACITY_DIR / "boekel" / "PAH.kappa", unpack=True)
OPTIONS.model.constant_params["pah"] = pah

tempc = Parameter(**STANDARD_PARAMETERS.temp0)
tempc.shortname = tempc.name = "tempc"
tempc.description = "The temperature of the black body"
tempc.value = 900

cont_weight = Parameter(**STANDARD_PARAMETERS.cont_weight)
cont_weight.set(min=0, max=1)
cont_weight.value = 0.5

pah_weight = Parameter(**STANDARD_PARAMETERS.cont_weight)
pah_weight.shortname = pah_weight.name = "pah_weight"
pah_weight.description = "The mass fraction for the PAHs"
pah_weight.set(min=0, max=5)
pah_weight.value = 0.25

factor = Parameter(**STANDARD_PARAMETERS.f)
factor.name = factor.shortname = "factor"
factor.description = "The factor to scale the black body"
factor.unit, factor.value = u.one, 19
factor.set(min=15, max=25)

sed = {"tempc": tempc, "pah_weight": pah_weight,
       "cont_weight": cont_weight, "factor": factor}
for key in NAMES.keys():
    for size in ["small", "large"]:
        weight_name = f"{key}_{size}_weight"
        weight = Parameter(**STANDARD_PARAMETERS.cont_weight)
        weight.shortname = weight.name = weight_name
        weight.description = f"The mass fraction for {size} {key}"
        weight.set(min=0, max=1)
        weight.value = 0.5
        sed[weight_name] = weight

OPTIONS.model.components_and_params = [["SED", sed]]
LABELS, UNITS = [key for key in sed], [value.unit for value in sed.values()]
component_labels = ["SED"]
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

model_flux = compute_sed(components, all_wavelengths)
chi_sq = compute_chi_sq(
    data.flux.value, data.flux.err, model_flux, func_method="default")

# NOTE: The +1 here is due to the fact that the last parameter is constrained
# by the others and not really fitted
nfree_params = len(get_priors()) + 1
rchi_sq = chi_sq / (data.flux.value.size - nfree_params)
print(f"rchi_sq: {rchi_sq:.2f}")


if __name__ == "__main__":
    ncores = 70
    fit_params = {"nlive_init": 2000, "lnprob": lnprob_sed, "ptform": ptform}
    sampler = run_fit(**fit_params, ncores=ncores,
                      method="dynamic", save_dir=result_dir,
                      debug=False)

    theta, uncertainties = get_best_fit(sampler, **fit_params)
    components_and_params, shared_params = set_params_from_theta(theta)
    components = basic_components.assemble_components(
            components_and_params, shared_params)
    model_flux = compute_sed(components, all_wavelengths)
    chi_sq = compute_chi_sq(
        data.flux.value, data.flux.err, model_flux, func_method="default")
    rchi_sq = chi_sq / (data.flux.value.size - nfree_params)
    print(f"rchi_sq: {rchi_sq:.2f}")

    save_fits(
        components, component_labels,
        fit_hyperparameters=fit_params, ncores=ncores,
        save_dir=result_dir / "model.fits",
        object_name="HD142527")

    plot_corner(sampler, LABELS, UNITS, savefig=result_dir / "corner.pdf")
    plot_sed([7.9, 13.3] * u.um, components, scaling="nu", save_dir=result_dir)
    plot_sed([7.9, 13.3] * u.um, components, scaling=None, save_dir=result_dir)

