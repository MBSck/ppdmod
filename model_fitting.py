from pathlib import Path

import astropy.units as u
import numpy as np

from ppdmod import data
from ppdmod import utils
from ppdmod.parameter import STANDARD_PARAMETERS, Parameter
from ppdmod.options import OPTIONS
from ppdmod.mcmc import run_mcmc


data.set_fit_wavelengths([8.28835527e-06])
print(OPTIONS["fit.wavelengths"])

PATH = Path("/Users/scheuck/Data/reduced_data/hd142666/matisse")
FILES = ["hd_142666_2022-04-23T03_05_25:2022-04-23T02_28_06_AQUARIUS_FINAL_TARGET_INT.fits"]
FILES = list(map(lambda x: PATH / x, FILES))
data.get_data(FILES)

# NOTE: Geometric parameters
FOV, PIXEL_SIZE = 100, 0.1
DIM = utils.get_next_power_of_two(FOV / PIXEL_SIZE)

# NOTE: Star parameters
DISTANCE = 150
EFF_TEMP = 7500
EFF_RADIUS = 1.8

# NOTE: Temperature gradient parameters
INNER_TEMP = 1500

# NOTE: Get the wavelength axis of MATISSE for both band (CHECK: Might differ for different files)?
WAVELENGTH_AXES = list(
    map(lambda x: data.ReadoutFits(PATH / x).wavelength, FILES))

RIN = Parameter(name="rin", value=0, unit=u.mas,
                description="Inner radius of the disk")
ROUT = Parameter(name="rout", value=0, unit=u.mas,
                 description="Outer radius of the disk")
MDUST = Parameter(name="Mdust", value=0.0, unit=u.M_sun,
                  description="Mass of the dusty disk")
P = Parameter(name="p", value=0, unit=u.one,
              description="Power-law exponent for the surface density profile")
A = Parameter(name="a", value=0, unit=u.one,
              description="Azimuthal modulation amplitude")
PHI = Parameter(name="phi", value=0, unit=u.deg,
                description="Azimuthal modulation angle")
CONT_WEIGHT = Parameter(name="cont_weight", value=0.0,
                        unit=u.one, free=True,
                        description="Dust mass continuum absorption coefficient's weight")
PA = Parameter(**STANDARD_PARAMETERS["pa"])
ELONG = Parameter(**STANDARD_PARAMETERS["elong"])

RIN.set(min=0, max=20)
ROUT.set(min=0, max=100)
MDUST.set(min=0, max=3)
P.set(min=0., max=1.)
A.set(min=0., max=1.)
PHI.set(min=0, max=360)
CONT_WEIGHT.set(min=0, max=1)
PA.set(min=0, max=360)
ELONG.set(min=1, max=50)

# NOTE: Wavelength dependent parameters
WEIGHTS = np.array([42.8, 9.7, 43.5, 1.1, 2.3, 0.6])/100
QVAL_FILE_DIR = Path("/Users/scheuck/Data/opacities/QVAL")
QVAL_FILES = ["Q_Am_Mgolivine_Jae_DHS_f1.0_rv0.1.dat",
              "Q_Am_Mgolivine_Jae_DHS_f1.0_rv1.5.dat",
              "Q_Am_Mgpyroxene_Dor_DHS_f1.0_rv1.5.dat",
              "Q_Fo_Suto_DHS_f1.0_rv0.1.dat",
              "Q_Fo_Suto_DHS_f1.0_rv1.5.dat",
              "Q_En_Jaeger_DHS_f1.0_rv1.5.dat"]
QVAL_PATHS = list(map(lambda x: QVAL_FILE_DIR / x, QVAL_FILES))
OPACITY_N_BAND = utils.linearly_combine_opacities(
    WEIGHTS, QVAL_PATHS, WAVELENGTH_AXES[0])
CONTINUUM_OPACITY_N_BAND = utils.opacity_to_matisse_opacity(
    WAVELENGTH_AXES[0], qval_file=QVAL_FILE_DIR / "Q_SILICA_RV0.1.DAT")

KAPPA_ABS = OPACITY_N_BAND
KAPPA_ABS_CONT = CONTINUUM_OPACITY_N_BAND

KAPPA_ABS = Parameter(name="kappa_abs", value=KAPPA_ABS,
                      wavelength=WAVELENGTH_AXES[0],
                      unit=u.cm**2/u.g, free=False,
                      description="Dust mass absorption coefficient")
KAPPA_ABS_CONT = Parameter(name="kappa_cont", value=KAPPA_ABS_CONT,
                           wavelength=WAVELENGTH_AXES[0],
                           unit=u.cm**2/u.g, free=False,
                           description="Continuum dust mass absorption coefficient")

OPTIONS["model.params"] = {"rin": RIN, "rout": ROUT,
                           "dust_mass": MDUST, "p": P,
                           "a": A, "phi": PHI,
                           "cont_weight": CONT_WEIGHT,
                           "pa": PA, "elong": ELONG}
OPTIONS["model.constant_params"] = {"dim": DIM, "pixel_size": PIXEL_SIZE,
                                    "dist": DISTANCE, "eff_temp": EFF_TEMP,
                                    "eff_radius": EFF_RADIUS,
                                    "inner_temp": INNER_TEMP,
                                    "kappa_abs": KAPPA_ABS,
                                    "kappa_cont": KAPPA_ABS_CONT}

if __name__ == "__main__":
    sampler = run_mcmc(25)
    breakpoint()
