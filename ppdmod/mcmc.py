from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Optional, Callable, Dict, List

import astropy.units as u
import emcee
import numpy as np
import matplotlib.pyplot as plt

from .custom_components import Star, AsymmetricSDGreyBodyContinuum
from .data import ReadoutFits
from .fft import interpolate_for_coordinates
from .model import Model
from .parameter import Parameter, STANDARD_PARAMETERS
from .utils import get_next_power_of_two, opacity_to_matisse_opacity,\
    linearly_combine_opacities, execution_time


# NOTE: Path to data.
PATH = Path("/Users/scheuck/Data/reduced_data/hd142666/matisse")

# NOTE: Define wavelengths to fit
WAVELENGTHS = np.array([4.78301581e-06, 8.28835527e-06])

# NOTE: Get the wavelenght axis of MATISSE for both band (CHECK: Might differ for different files)?
FILES = ["hd_142666_2022-04-21T07_18_22:2022-04-21T06_47_05_HAWAII-2RG_FINAL_TARGET_INT.fits",
         "hd_142666_2022-04-23T03_05_25:2022-04-23T02_28_06_AQUARIUS_FINAL_TARGET_INT.fits"]
WAVELENGTH_AXES = list(map(lambda x: ReadoutFits(PATH / x).wavelength, FILES))
WAVELENGTH_AXES[0].sort()
WAVELENGTH_AXIS = np.concatenate(WAVELENGTH_AXES)

FILES = list(map(lambda x: PATH / x, FILES))
DATA = [ReadoutFits(file) for file in FILES]
CORR_FLUX, CORR_FLUX_ERR = [], []
CPHASE, CPHASE_ERR = [], []

for data in DATA:
    CORR_FLUX.append(data.get_data_for_wavelength(WAVELENGTHS, "vis"))
    CORR_FLUX_ERR.append(data.get_data_for_wavelength(WAVELENGTHS, "vis_err"))
    CPHASE.append(data.get_data_for_wavelength(WAVELENGTHS, "t3phi"))
    CPHASE_ERR.append(data.get_data_for_wavelength(WAVELENGTHS, "t3phi_err"))

# NOTE: Geometric parameters
FOV, PIXEL_SIZE = 100, 0.1
DIM = get_next_power_of_two(FOV / PIXEL_SIZE)

# NOTE: Star parameters
DISTANCE = 150
EFF_TEMP = 7500
EFF_RADIUS = 1.8
LUMINOSITY = 9.12

# NOTE: Temperature gradient parameters
INNER_TEMP = 1500

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
OPACITY_L_BAND = linearly_combine_opacities(WEIGHTS,
                                            QVAL_PATHS, WAVELENGTH_AXES[0])
OPACITY_N_BAND = linearly_combine_opacities(WEIGHTS,
                                            QVAL_PATHS, WAVELENGTH_AXES[1])
CONTINUUM_OPACITY_L_BAND = opacity_to_matisse_opacity(WAVELENGTH_AXES[0],
                                                      qval_file=QVAL_FILE_DIR / "Q_SILICA_RV0.1.DAT")
CONTINUUM_OPACITY_N_BAND = opacity_to_matisse_opacity(WAVELENGTH_AXES[1],
                                                      qval_file=QVAL_FILE_DIR / "Q_SILICA_RV0.1.DAT")

KAPPA_ABS = np.concatenate([OPACITY_L_BAND, OPACITY_N_BAND])
KAPPA_ABS_CONT = np.concatenate([CONTINUUM_OPACITY_L_BAND,
                                 CONTINUUM_OPACITY_N_BAND])

KAPPA_ABS = Parameter(name="kappa_abs", value=KAPPA_ABS,
                      wavelengths=WAVELENGTH_AXIS,
                      unit=u.cm**2/u.g, free=False,
                      description="Dust mass absorption coefficient")
KAPPA_ABS_CONT = Parameter(name="kappa_cont", value=KAPPA_ABS_CONT,
                           wavelengths=WAVELENGTH_AXIS,
                           unit=u.cm**2/u.g, free=False,
                           description="Continuum dust mass absorption coefficient")

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

PARAMS = {"rin": RIN, "rout": ROUT, "Mdust": MDUST,
          "p": P, "a": A, "phi": PHI, "cont_weight": CONT_WEIGHT,
          "pa": PA, "elong": ELONG}


def chi_sq(data: u.quantity, error: u.quantity,
           model_data: u.quantity, lnf: float = 0) -> float:
    """the chi square minimisation.

    Parameters
    ----------
    data : numpy.ndarray
    error : numpy.ndarray
    model_data : numpy.ndarray
    lnf : float, optional

    Returns
    -------
    chi_sq : float
    """
    inv_sigma_squared = 1./np.sum(error**2
                                  + model_data**2*np.exp(2*lnf))
    return -0.5*np.sum((data-model_data)**2*inv_sigma_squared
                       - np.log(inv_sigma_squared))

# @execution_time
def lnprob(theta: np.ndarray) -> float:
    """Takes theta vector and the x, y and the yerr of the theta.
    Returns a number corresponding to how good of a fit the model is to your
    data for a given set of parameters, weighted by the data points.


    Parameters
    ----------
    theta: np.ndarray
        A list of all the parameters that ought to be fitted
    data: DataHandler

    Returns
    -------
    float
        The goodness of the fitted model (will be minimised)

    Notes
    -----
    If multiple models with the same params are required put them once into theta
    and then assign them with a dict, to then get them from the dict for the second
    component as well.
    """
    params = dict(zip(PARAMS.keys(), theta))
    # TODO: If more than one model component make all params variable
    for index, parameter in enumerate(PARAMS.values()):
        if parameter.free:
            if not parameter.min < theta[index] < parameter.max:
                return -np.inf

    star = Star(dim=DIM, dist=DISTANCE, eff_temp=EFF_TEMP, lum=LUMINOSITY)
    temp_grad = AsymmetricSDGreyBodyContinuum(dim=DIM, dist=DISTANCE,
                                              Tin=INNER_TEMP,
                                              pixSize=PIXEL_SIZE,
                                              Teff=EFF_TEMP,
                                              kappa_abs=KAPPA_ABS,
                                              lum=LUMINOSITY,
                                              kappa_cont=KAPPA_ABS_CONT,
                                              **params)
    pixel_size = temp_grad.params["pixSize"].value\
        * temp_grad.params["pixSize"].unit.to(u.rad)
    model = Model([star, temp_grad])

    # TODO: Apply coordinate transformation somehow?
    fourier_transforms = {}
    for wavelength in WAVELENGTHS:
        fourier_transform = model.calculate_complex_visibility(wavelength)
        # total_flux = fourier_transform
        fourier_transforms[str(wavelength)] = {"ft": fourier_transform,
                                               "total_flux": total_flux}

    # TODO: Calculate the different observables here
    total_chi_sq = 0
    for index, (corr_flux, corr_flux_err, cphase, cphase_err)\
            in enumerate(zip(CORR_FLUX, CORR_FLUX_ERR, CPHASE, CPHASE_ERR)):
        print(index)
        for wavelength in WAVELENGTHS:
            print(wavelength)
            if str(wavelength) in fourier_transforms\
                    and str(wavelength) in corr_flux:
                fourier_transform = fourier_transforms[str(wavelength)]
                interpolated_fft = interpolate_for_coordinates(
                    fourier_transform["ft"],
                    fourier_transform["ft"].shape[0],
                    pixel_size,
                    DATA[index].ucoord, DATA[index].vcoord,
                    DATA[index].wavelength, wavelength)
                # total_flux_model = fourier_transform["total_flux"]
                corr_flux_model = np.abs(interpolated_fft)
                total_chi_sq += chi_sq(corr_flux[str(wavelength)],
                                       corr_flux_err[str(wavelength)],
                                       corr_flux_model)
                # cphase_model = ...
                # total_chi_sq += chi_sq(cphase, cphase_err, cphase_model)
            else:
                continue
    return np.array(total_chi_sq)


def initiate_randomly(free_parameters: Dict[str, Parameter],
                      nwalkers: float) -> np.ndarray:
    """initialises a random numpy.ndarray from the parameter's limits.

    parameters
    -----------
    free_parameters : dict of Parameter

    Returns
    -------
    numpy.ndarray
    """
    initial = np.ndarray([nwalkers, len(free_parameters)])
    for index, parameter in enumerate(free_parameters.values()):
        initial[:, index] = np.random.random(nwalkers)\
            * np.ptp([parameter.min, parameter.max])+parameter.min
    return initial


def run_mcmc(nwalkers: int,
             nsteps: int,
             discard: int,
             save_path: Optional[Path] = None) -> np.array:
    """Runs the emcee Hastings Metropolitan sampler

    The EnsambleSampler recieves the parameters and the args are passed to
    the 'log_prob()' method (an addtional parameter 'a' can be used to
    determine the stepsize, defaults to None).

    The burn-in is first run to explore the parameter space and then the
    walkers settle into the maximum of the density. The state variable is
    then passed to the production run.

    The chain is reset before the production with the state variable being
    passed. 'rstate0' is the state of the internal random number generator

    Parameters
    ----------
    parameters : dict of Parameter
    nwalkers : int
    nsteps : int
    discard : int
    wavelengths : float
    save_path: pathlib.Path, optional

    Returns
    -------
    np.ndarray
    """
    initial = initiate_randomly(PARAMS, nwalkers)
    with Pool() as pool:
        print(f"Executing MCMC with {cpu_count()} cores.")
        print("--------------------------------------------------------------")
        sampler = emcee.EnsembleSampler(nwalkers, len(PARAMS),
                                        lnprob, pool=pool)
        sampler.run_mcmc(initial, nsteps, progress=True)
    return (sampler.flatchain)[np.argmax(sampler.flatlnprobability)]


if __name__ == "__main__":
    # OPTIONS["fourier.binning"] = 1
    # star = Star(dim=DIM, dist=DISTANCE, eff_temp=EFF_TEMP, lum=LUMINOSITY)
    # temp_grad = AsymmetricSDGreyBodyContinuum(dim=DIM, dist=DISTANCE, Tin=INNER_TEMP,
    #                                           pixSize=PIXEL_SIZE, Teff=EFF_TEMP,
    #                                           kappa_abs=KAPPA_ABS, lum=LUMINOSITY,
    #                                           kappa_cont=KAPPA_ABS_CONT,
    #                                           rin=0.5, rout=100, Mdust=0.11,
    #                                           q=0.5, p=0.5, a=0.5, cont_weight=0.5,
    #                                           pa=150, phi=33, elong=0.5)
    # model = Model([star, temp_grad])
    # model.plot_model(4096, 0.1, 4.78301581e-06)
    # plt.show()
    # run_mcmc(nwalkers=25, nsteps=100, discard=10)
    initial = initiate_randomly(PARAMS, 25)
    lnprob(initial[0])
