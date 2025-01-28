import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import List

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import astropy.units as u
import numpy as np

from ppdmod.basic_components import AsymGreyBody, GreyBody, Star
from ppdmod.data import set_data
from ppdmod.fitting import (
    compute_interferometric_chi_sq,
    get_best_fit,
    ptform_sequential_radii,
    run_fit,
    set_components_from_theta,
)
from ppdmod.options import OPTIONS
from ppdmod.parameter import Parameter
from ppdmod.utils import (
    load_data,
    qval_to_opacity,
    windowed_linspace,
    create_adaptive_bins,
)


DATA_DIR = Path(__file__).parent.parent / "data"
nband_wavelengths, nband_binning_windows = create_adaptive_bins([8.6, 12.3], [9.2, 11.9], 0.2, 0.65)
wavelengths = {
    "hband": [1.7] * u.um,
    "kband": [2.15] * u.um,
    "lband": windowed_linspace(3.1, 3.8, OPTIONS.data.binning.lband.value) * u.um,
    "mband": windowed_linspace(4.65, 4.9, OPTIONS.data.binning.mband.value) * u.um,
    "nband": nband_wavelengths * u.um,
}
OPTIONS.data.binning.nband = nband_binning_windows * u.um
fits_files = list((DATA_DIR / "fits" / "hd142527").glob("*fits"))
bands = ["hband", "kband", "lband", "mband", "nband"]
wavelengths = np.concatenate([wavelengths[band] for band in bands])

# fit_data = ["flux", "vis", "t3"]
fit_data = ["flux", "vis"]
data = set_data(
    fits_files,
    wavelengths=wavelengths,
    fit_data=fit_data,
    average=True,
)

grid, value = np.loadtxt(
    DATA_DIR / "flux" / "hd142527" / "HD142527_stellar_model.txt",
    usecols=(0, 2),
    unpack=True,
)
flux_star = Parameter(grid=grid, value=value, base="f")

SOURCE_DIR = DATA_DIR / "results" / "hd142527"

method = "grf"
grid, value = np.load(SOURCE_DIR / f"silicate_{method}_opacities.npy")
kappa_abs = Parameter(grid=grid, value=value, base="kappa_abs")

grid, value = load_data(
    DATA_DIR / "opacities" / "qval" / "Q_amorph_c_rv0.1.dat", load_func=qval_to_opacity
)
kappa_cont = Parameter(grid=grid, value=value, base="kappa_cont")
pa = Parameter(value=352, free=False, base="pa")
cinc = Parameter(value=0.84, free=False, base="cinc")

with open(SOURCE_DIR / "opacity_temps.pkl", "rb") as save_file:
    temps = pickle.load(save_file)

x = Parameter(free=True, base="x")
y = Parameter(free=True, base="y")

rin1 = Parameter(value=0.1, min=0, max=2, unit=u.au, base="rin")
rout1 = Parameter(value=1.5, min=0, max=2, unit=u.au, free=True, base="rout")
p1 = Parameter(value=0.5, min=-10, max=10, base="p")
sigma01 = Parameter(value=1e-3, min=0, max=1e-1, base="sigma0")

rin2 = Parameter(value=2, min=1, max=9, unit=u.au, base="rin")
rout2 = Parameter(value=4, min=3, max=45, unit=u.au, free=True, base="rout")
p2 = Parameter(value=0.5, min=-30, max=20, base="p")
sigma02 = Parameter(value=1e-3, min=0, max=1e-1, base="sigma0")
rho21 = Parameter(value=0.6, free=True, base="rho")
theta21 = Parameter(value=33, free=True, base="theta")

flux_lnf = Parameter(name="flux_lnf", free=True, base="lnf")
vis_lnf = Parameter(name="vis_lnf", free=True, base="lnf")
t3_lnf = Parameter(name="t3_lnf", free=True, base="lnf")

shared_params = {
    "dim": 32,
    "dist": 158.51,
    "eff_temp": 6750,
    "eff_radius": 3.46,
    "kappa_abs": kappa_abs,
    "kappa_cont": kappa_cont,
    "pa": pa,
    "cinc": cinc,
    # "flux_lnf": flux_lnf,
    # "vis_lnf": vis_lnf,
    # "t3_lnf": t3_lnf,
    # "weights": temps.weights,
    # "radii": temps.radii,
    # "matrix": temps.values,
}

star = Star(label="Star", f=flux_star, **shared_params)
inner_ring = GreyBody(
    label="Inner Ring",
    rin=rin1,
    rout=rout1,
    p=p1,
    sigma0=sigma01,
    **shared_params,
)
outer_ring = AsymGreyBody(
    label="Outer Ring",
    rin=rin2,
    rout=rout2,
    p=p2,
    sigma0=sigma02,
    rho1=rho21,
    theta1=theta21,
    **shared_params,
)

OPTIONS.model.components = components = [star, inner_ring, outer_ring]
DIR_NAME = "better_constraints"
if DIR_NAME is None:
    DIR_NAME = f"results_model_{datetime.now().strftime('%H:%M:%S')}"

result_dir = DATA_DIR.parent / "results" / "disc_fit"
day_dir = Path(str(datetime.now().date()))
result_dir /= day_dir / DIR_NAME
result_dir.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ncores = 70
    OPTIONS.fit.condition = "sequential_radii"
    fit_params = {"dlogz_init": 0.01, "nlive_init": 4000, "nlive_batch": 1000}
    sampler = run_fit(**fit_params, ncores=ncores, save_dir=result_dir, debug=False)
    theta, uncertainties = get_best_fit(sampler)
    components = OPTIONS.model.components = set_components_from_theta(theta)
    np.save(result_dir / "uncertainties.npy", uncertainties)

    with open(result_dir / "components.pkl", "wb") as file:
        pickle.dump(components, file)

    rchi_sq = compute_interferometric_chi_sq(
        components,
        theta.size,
        method="linear",
        reduced=True,
    )[0]
    print(f"Total reduced chi_sq: {rchi_sq:.2f}")
