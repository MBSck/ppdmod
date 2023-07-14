from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Optional, Callable, Dict

import astropy.units as u
import emcee
import numpy as np

from custom_components import Star, TemperatureGradient
from data import ReadoutFits
from model import Model
from parameter import Parameter
from utils import get_next_power_of_two, opacity_to_matisse_opacity,\
    linearly_combine_opacities


# NOTE: Path to data.
PATH = Path("/Users/scheuck/Data/reduced_data/hd142666/matisse")

# NOTE: Define wavelengths to fit
WAVELENGTHS = []

# NOTE: Get the wavelenght axis of MATISSE for both band (CHECK: Might differ for different files)?
WAVLENGTH_FILES = ["hd_142666_2022-04-21T07_18_22:2022-04-21T06_47_05_HAWAII-2RG_FINAL_TARGET_INT.fits",
                   "hd_142666_2022-04-21T07_18_22:2022-04-21T06_47_05_AQUARIUS_FINAL_TARGET_INT.fits"]
WAVELENGTH_AXES = list(map(lambda x: ReadoutFits(PATH / x).wavelength, WAVLENGTH_FILES))
WAVELENGTH_AXES[0].sort()
WAVELENGTH_AXIS = np.concatenate(WAVELENGTH_AXES)

# Define globals
FILES = ["hd_142666_2022-04-21T07_18_22:2022-04-21T06_47_05_AQUARIUS_FINAL_TARGET_INT.fits",
         "hd_142666_2022-04-23T03_05_25:2022-04-23T02_28_06_AQUARIUS_FINAL_TARGET_INT.fits"]
FILES = list(map(lambda x: PATH / x, FILES))
DATA = [ReadoutFits(file) for file in FILES]

# NOTE: Geometric parameters
FOV, PIXEL_SIZE = 100, 0.1
DIM = get_next_power_of_two(FOV / PIXEL_SIZE)

# NOTE: Star parameters
DISTANCE = 150
EFF_TEMP = 7500
LUMINOSITY = 19

# NOTE: Temperature gradient parameters
INNER_TEMP = 1500

# NOTE: Wavelength dependent parameters
WEIGHTS = np.array([42.8, 9.7, 43.5, 1.1, 2.3, 0.6])/100
QVAL_FILE_DIR = Path("/Users/scheuck/Data/opacities/QVAL")
QVAL_FILES = ["Q_Am_Mgolivine_Jae_DHS_f0.2_rv0.1.dat",
              "Q_Am_Mgolivine_Jae_DHS_f0.2_rv1.5.dat",
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


def chi_sq(data: u.quantity, error: u.quantity,
           model_data: u.quantity, lnf: float = 0) -> float:
    """the chi square minimisation.

    Parameters
    ----------
    data : astropy.units.Quantity
    error : astropy.units.Quantity
    model_data : astropy.units.Quantity
    lnf : float, optional

    Returns
    -------
    chi_sq : float
    """
    inv_sigma_squared = 1./np.sum(error.value**2+\
                                  model_data.value**2*np.exp(2*lnf))
    return -0.5*np.sum((data.value-model_data.value)**2*inv_sigma_squared\
                       - np.log(inv_sigma_squared))

# TODO: Think of a way to handle data here.
def lnlike(theta: np.ndarray,
           wavelengths: np.ndarray) -> float:
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
    """
    # TODO: Initialise model here to account for multiprocessing
    # TODO: Set params globally and then just pass them to the function and set the values from the theta.
    star = Star(dim=DIM, dist=DISTANCE, eff_temp=EFF_TEMP, lum=LUMINOSITY)
    temp_grad = TemperatureGradient(dim=DIM, )
    model = Model([star, temp_grad])
    fourier_transforms = {}
    # NOTE: Find way to check if file has wavelength and then use the uv-coords of the wavelength.
    for wavelength in WAVELENGTHS:
        fourier_transform = model.calculate_complex_visibility(DATA.ucoord, DATA.vcoord,
                                                               WAVELENGTHS)
    # TODO: Calculate the different observables here
    total = 0
    for index, _ in enumerate(FILES):
        data = DATA[index].get_data_for_wavelength(WAVELENGTHS, key)
        error = DATA[index].get_data_for_wavelength(WAVELENGTHS, f"{key}_err")
        total += chi_sq(data, error, model)
    return np.array(total)


def lnprior(parameters: Dict[str, Parameter]) -> float:
    """Checks if all variables are within their priors (as well as
    determining them setting the same).

    If all priors are satisfied it needs to return '0.0' and if not '-np.inf'
    This function checks for an unspecified amount of flat priors. If upper
    bound is 'None' then no upper bound is given

    Parameters
    ----------
    parameters : dict of Parameter

    Returns
    -------
    float
        Return-code 0.0 for within bounds and -np.inf for out of bound priors
    """
    for parameter in parameters:
        if parameter.free:
            if not parameter.min < parameter.value < parameter.max:
                return -np.inf
    return 0.

def lnprob(parameters: Dict[str, Parameter]) -> np.ndarray:
    """This function runs the lnprior and checks if it returned -np.inf, and
    returns if it does. If not, (all priors are good) it returns the inlike for
    that model (convention is lnprior + lnlike)

    Parameters
    ----------
    parameters : dict of Parameter
        A vector that contains all the parameters of the model

    Returns
    -------
    float
        The minimisation value or -np.inf if it fails
    """
    return lnlike(parameters) if np.isfinite(lnprior(parameters)) else -np.inf


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


def run_mcmc(parameters: Dict[str, Parameter],
             nwalkers: int,
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
    nwalkers : int,
    nsteps : int,
    discard : int,
    wavelengths : float
    save_path: pathlib.Path, optional

    Returns
    -------
    np.ndarray
    """
    inital = initiate_randomly(parameters, nwalkers)
    with Pool() as pool:
        print(f"Executing MCMC with {cpu_count()} cores.")
        print("--------------------------------------------------------------")
        sampler = emcee.EnsembleSampler(nwalkers, len(MODEL.free_params), lnprob,
                                        args=(parameterswavelengths), pool=pool)
        pos, prob, state = sampler.run_mcmc(inital, nsteps, progress=True)
    theta_max = (sampler.flatchain)[np.argmax(sampler.flatlnprobability)]


if __name__ == "__main__":
    star = Star(dim=DIM, dist=DISTANCE, eff_temp=EFF_TEMP, lum=LUMINOSITY)
    temp_grad = TemperatureGradient(dim=DIM, dist=DISTANCE, Tin=INNER_TEMP,
                                    pixSize=PIXEL_SIZE, rin=0.5, rout=100,
                                    Mdust=0.11, q=0.5, p=0.5, a=0.5,
                                    pa=150, phi=33, elong=0.5)
    model = Model([star, temp_grad])
    model.calculate_image(1024, 0.1, 3.5e-6)
    breakpoint()
