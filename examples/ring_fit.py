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

from ppdmod.basic_components import Ring
from ppdmod.data import set_data
from ppdmod.fitting import (
    compute_interferometric_chi_sq,
    compute_observables,
    get_best_fit,
    get_labels,
    run_fit,
    set_components_from_theta,
)
from ppdmod.options import OPTIONS
from ppdmod.parameter import Parameter


DATA_DIR = Path(__file__).parent.parent / "data"
wavelengths = {
    "hband": [1.7] * u.um,
    "kband": [2.15] * u.um,
    "lband": np.linspace(3.1, 3.4, 6) * u.um,
    "mband": np.linspace(4.7, 4.9, 4) * u.um,
    "nband": np.linspace(8, 15, 35) * u.um,
}

fits_files = list((DATA_DIR / "fits" / "hd142527").glob("*fits"))
wavelengths = np.concatenate(
    (
        # wavelengths["hband"],
        wavelengths["kband"],
        # wavelengths["lband"],
        # wavelengths["mband"],
        # wavelengths["nband"],
    )
)
data = set_data(
    fits_files,
    wavelengths=wavelengths,
    fit_data=["vis2"],
)

SOURCE_DIR = DATA_DIR / "model_results" / "hd142527"

rin = Parameter(value=0.95308189, min=0, max=3, base="rin")
pa = Parameter(value=352, free=False, base="pa")
cinc = Parameter(value=1.0, free=True, base="cinc")
ring = Ring(rin=rin, pa=pa, cinc=cinc)

OPTIONS.model.components = components = [ring]
LABELS = get_labels(components)

result_dir = Path("../model_results/") / "disc_fit"
day_dir = result_dir / str(datetime.now().date())
dir_name = f"results_model_{datetime.now().strftime('%H:%M:%S')}"
result_dir = day_dir / dir_name
result_dir.mkdir(parents=True, exist_ok=True)

ndim = len(LABELS)
rchi_sqs = compute_interferometric_chi_sq(
    *compute_observables(components),
    ndim=ndim,
    method="linear",
    reduced=True,
)
print(f"rchi_sq: {rchi_sqs[0]:.2f}")


if __name__ == "__main__":
    ncores = 50
    fit_params = {"nlive_init": 2000, "batch_size": 1000}
    sampler = run_fit(ncores=ncores, save_dir=result_dir, debug=False, **fit_params)

    theta, uncertainties = get_best_fit(sampler)
    OPTIONS.model.components = components = set_components_from_theta(theta)
    np.save(result_dir / "uncertainties.npy", uncertainties)

    with open(result_dir / "components.pkl", "wb") as file:
        pickle.dump(components, file)

    rchi_sqs = compute_interferometric_chi_sq(
        *compute_observables(components),
        ndim=ndim,
        method="linear",
        reduced=True,
    )
    print(f"Total reduced chi_sq: {rchi_sqs[0]:.2f}")
