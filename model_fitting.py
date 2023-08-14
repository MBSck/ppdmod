from pathlib import Path

import astropy.units as u
import numpy as np

from ppdmod import data
from ppdmod import mcmc
from ppdmod import plot
from ppdmod import utils
from ppdmod.parameter import STANDARD_PARAMETERS, Parameter
from ppdmod.options import OPTIONS


data.set_fit_wavelengths([8.28835527e-06])
path = Path("/Users/scheuck/Data/reduced_data/hd142666/matisse")
files = ["hd_142666_2022-04-23T03_05_25:2022-04-23T02_28_06_AQUARIUS_FINAL_TARGET_INT.fits"]
files = list(map(lambda x: path / x, files))
data.get_data(files)

fov, pixel_size = 100, 0.1
dim = utils.get_next_power_of_two(fov / pixel_size)
const_params = {"dist": 150, "eff_temp": 7500,
                "pixel_size": 0.1,
                "eff_radius": 1.8, "inner_temp": 1500}

wavelength_axes = list(
    map(lambda x: data.ReadoutFits(path / x).wavelength, files))

rin = Parameter(**STANDARD_PARAMETERS["rin"])
rout = Parameter(**STANDARD_PARAMETERS["rout"])
inner_sigma = Parameter(**STANDARD_PARAMETERS["inner_sigma"])
p = Parameter(**STANDARD_PARAMETERS["p"])
a = Parameter(**STANDARD_PARAMETERS["a"])
phi = Parameter(**STANDARD_PARAMETERS["phi"])
cont_weight = Parameter(**STANDARD_PARAMETERS["cont_weight"])
pa = Parameter(**STANDARD_PARAMETERS["pa"])
elong = Parameter(**STANDARD_PARAMETERS["elong"])

rin.set(min=0, max=20)
rout.set(min=0, max=100)
p.set(min=0., max=1.)
a.set(min=0., max=1.)
phi.set(min=0, max=360)
cont_weight.set(min=0, max=1)
pa.set(min=0, max=360)
elong.set(min=1, max=50)

weights = np.array([42.8, 9.7, 43.5, 1.1, 2.3, 0.6])/100
qval_file_dir = Path("/Users/scheuck/Data/opacities/QVAL")
qval_files = ["Q_Am_Mgolivine_Jae_DHS_f1.0_rv0.1.dat",
              "Q_Am_Mgolivine_Jae_DHS_f1.0_rv1.5.dat",
              "Q_Am_Mgpyroxene_Dor_DHS_f1.0_rv1.5.dat",
              "Q_Fo_Suto_DHS_f1.0_rv0.1.dat",
              "Q_Fo_Suto_DHS_f1.0_rv1.5.dat",
              "Q_En_Jaeger_DHS_f1.0_rv1.5.dat"]
qval_paths = list(map(lambda x: qval_file_dir / x, qval_files))
opacity_n_band = utils.linearly_combine_opacities(
    weights, qval_paths, wavelength_axes[0])
continuum_opacity_n_band = utils.opacity_to_matisse_opacity(
    wavelength_axes[0], qval_file=qval_file_dir / "Q_SILICA_RV0.1.DAT")

kappa_abs = opacity_n_band
kappa_abs_cont = continuum_opacity_n_band

kappa_abs = Parameter(name="kappa_abs", value=kappa_abs,
                      wavelength=wavelength_axes[0],
                      unit=u.cm**2/u.g, free=False,
                      description="Dust mass absorption coefficient")
kappa_abs_cont = Parameter(name="kappa_cont", value=kappa_abs_cont,
                           wavelength=wavelength_axes[0],
                           unit=u.cm**2/u.g, free=False,
                           description="Continuum dust mass absorption coefficient")


if __name__ == "__main__":
    sampler = mcmc.run_mcmc(25)
    theta = mcmc.get_best_fit(sampler)
    plot.plot_chains(theta)
    breakpoint()
