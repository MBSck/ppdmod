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

from ppdmod.components import AsymTempGrad, Point
from ppdmod.data import set_data
from ppdmod.fitting import (
    compute_interferometric_chi_sq,
    get_best_fit,
    get_labels,
    run_fit,
    set_components_from_theta,
)
from ppdmod.options import OPTIONS
from ppdmod.parameter import Parameter
from ppdmod.utils import (
    load_data,
    qval_to_opacity,
    windowed_linspace,
)

DATA_DIR = Path(__file__).parent.parent / "data"
RESULT_DIR_NAME = "all_data"
if RESULT_DIR_NAME is None:
    RESULT_DIR_NAME = f"results_model_{datetime.now().strftime('%H:%M:%S')}"

RESULT_DIR = DATA_DIR.parent / "results" / "disc_fit"
DAY_DIR = Path(str(datetime.now().date()))
RESULT_DIR /= DAY_DIR / RESULT_DIR_NAME
RESULT_DIR.mkdir(parents=True, exist_ok=True)

wavelengths = {
    "hband": [1.7] * u.um,
    "kband": [2.15] * u.um,
    "lband": windowed_linspace(3, 4, 0.2) * u.um,
    "mband": windowed_linspace(4.6, 5, 0.2) * u.um,
    "nband": windowed_linspace(8, 13, 0.2) * u.um,
}
# bands = ["hband", "kband", "lband", "mband", "nband"]
bands = ["lband", "mband", "nband"]
# bands = ["nband"]
wavelengths = np.concatenate([wavelengths[band] for band in bands])

fit_data = ["flux", "vis", "t3"]
fits_files = list((DATA_DIR / "fits" / "hd142527").glob("*fits"))
data = set_data(
    fits_files,
    wavelengths=wavelengths,
    fit_data=fit_data,
)

np.save(RESULT_DIR / "wl.npy", wavelengths.value)
np.save(RESULT_DIR / "observables.npy", fit_data)
np.save(RESULT_DIR / "files.npy", [fits_file.name for fits_file in fits_files])

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

x = Parameter(free=False, base="x")
y = Parameter(free=False, base="y")

rin1 = Parameter(value=0.1, min=0, max=4, unit=u.au, base="rin")
rout1 = Parameter(value=1.5, min=0, max=4, unit=u.au, free=True, base="rout")
p1 = Parameter(value=0, base="p")
sigma01 = Parameter(value=1e-3, base="sigma0")
rho11 = Parameter(value=0.6, free=True, base="rho")
theta11 = Parameter(value=33, free=True, base="theta")

rin2 = Parameter(value=0, min=0.2, max=7, unit=u.au, base="rin")
rout2 = Parameter(value=0, min=1.5, max=10, unit=u.au, free=True, base="rout")
p2 = Parameter(value=0, base="p")
sigma02 = Parameter(value=1e-3, base="sigma0")
rho21 = Parameter(value=0.6, free=True, base="rho")
theta21 = Parameter(value=33, free=True, base="theta")

rin3 = Parameter(value=0, min=1, max=10, unit=u.au, base="rin")
rout3 = Parameter(value=0, min=2, max=15, unit=u.au, free=True, base="rout")
p3 = Parameter(value=0, base="p")
sigma03 = Parameter(value=1e-3, base="sigma0")
rho31 = Parameter(value=0.6, free=True, base="rho")
theta31 = Parameter(value=33, free=True, base="theta")

pa = Parameter(value=352, free=False, shared=True, base="pa")
cinc = Parameter(value=0.84, free=True, shared=True, base="cinc")
q = Parameter(value=0.5, free=True, shared=True, base="q")
temp0 = Parameter(value=1500, free=True, shared=True, base="temp0")
flux_lnf = Parameter(name="flux_lnf", free=True, shared=True, base="lnf")
vis_lnf = Parameter(name="vis_lnf", free=True, shared=True, base="lnf")
t3_lnf = Parameter(name="t3_lnf", free=True, shared=True, base="lnf")

shared_params = {
    "dim": 32,
    "dist": 158.51,
    "eff_temp": 6750,
    "eff_radius": 3.46,
    "kappa_abs": kappa_abs,
    "kappa_cont": kappa_cont,
    "pa": pa,
    "cinc": cinc,
    "q": q,
    "temp0": temp0,
    # "flux_lnf": flux_lnf,
    # "vis_lnf": vis_lnf,
    # "t3_lnf": t3_lnf,
}

star = Point(label="Star", fr=flux_star, x=x, y=y, **shared_params)
first = AsymTempGrad(
    label="Inner Ring",
    rin=rin1,
    rout=rout1,
    p=p1,
    sigma0=sigma01,
    rho1=rho11,
    theta1=theta11,
    **shared_params,
)
second = AsymTempGrad(
    label="Outer Ring",
    rin=rin2,
    rout=rout2,
    p=p2,
    sigma0=sigma02,
    rho1=rho21,
    theta1=theta21,
    **shared_params,
)
third = AsymTempGrad(
    label="Outer Ring",
    rin=rin3,
    rout=rout3,
    p=p3,
    sigma0=sigma03,
    rho1=rho31,
    theta1=theta31,
    **shared_params,
)

OPTIONS.model.components = components = [star, first, second, third]

if __name__ == "__main__":
    labels = get_labels(components)
    OPTIONS.fit.fitter = "dynesty"
    OPTIONS.fit.condition = "sequential_radii"
    OPTIONS.fit.condition_indices = list(
        map(labels.index, (filter(lambda x: "rin" in x or "rout" in x, labels)))
    )
    fit_params = {"dlogz_init": 0.01, "nlive_init": 1500, "nlive_batch": 150}
    ncores = fit_params.get("nwalkers", 150) // 2
    sampler = run_fit(**fit_params, ncores=ncores, save_dir=RESULT_DIR, debug=False)
    theta, uncertainties = get_best_fit(sampler, discard=fit_params.get("discard", 0))
    components = OPTIONS.model.components = set_components_from_theta(theta)
    np.save(RESULT_DIR / "theta.npy", theta)
    np.save(RESULT_DIR / "uncertainties.npy", uncertainties)

    with open(RESULT_DIR / "components.pkl", "wb") as file:
        pickle.dump(components, file)

    rchi_sq = compute_interferometric_chi_sq(
        components,
        theta.size,
        method="linear",
        reduced=True,
    )[0]
    print(f"Total reduced chi_sq: {rchi_sq:.2f}")
    with open(RESULT_DIR / "chi_sq.txt", "wb") as file:
        file.write(f"Total reduced chi_sq: {rchi_sq:.2f}".encode())
