from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Optional, Callable, Dict, List

import astropy.units as u
import emcee
import numpy as np

from data import ReadoutFits
from parameter import Parameter
from options import OPTIONS
from utils import get_next_power_of_two

# Define globals
PATH = Path("/Users/scheuck/Data/reduced_data/hd142666/matisse")
FILES = ["hd_142666_2022-04-21T07_18_22:2022-04-21T06_47_05_AQUARIUS_FINAL_TARGET_INT.fits",
         "hd_142666_2022-04-23T03_05_25:2022-04-23T02_28_06_AQUARIUS_FINAL_TARGET_INT.fits"]
FILES = [PATH / file for file in FILES]
DATA = [ReadoutFits(file) for file in FILES]

# NOTE: Geometric parameters
FOV = 100
PIXEL_SIZE = 0.1
DIM = get_next_power_of_two(FOV / PIXEL_SIZE)

# NOTE: Star parameters and model component
DISTANCE = 150
LUMINOSITY = 19
EFFECTIVE_TEMPERATURE = 7500
STAR = ...

# NOTE: Wavelength dependent parameters
KAPPA_ABS = ...
KAPPA_ABS_CONT = ...


def chi_sq(data: u.quantity, error: u.quantity,
           model_data: u.quantity, lnf: float) -> float:
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
    model = ...
    fourier_transform = model.calculate_complex_visibility(DATA.ucoord, DATA.vcoord,
                                                           wavelengths)
    # TODO: Calculate the different observables here
    total = 0
    for key in OPTIONS["fit.datasets"]:
        data = DATA.get_data_for_wavelength(wavelengths, key)
        error = DATA.get_data_for_wavelength(wavelengths, f"key_{err}")
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

def lnprob(parameters: Dict[Parameter], wavelengths: np.ndarray) -> np.ndarray:
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
    return lnlike(parameters, wavelengths)\
        if np.isfinite(lnprior(parameters)) else -np.inf


def initiate_randomly(free_parameters: Dict[Parameter],
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
             wavelengths: float,
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
                                        args=(parameters, wavelengths), pool=pool)
        pos, prob, state = sampler.run_mcmc(inital, nsteps, progress=True)
    theta_max = (sampler.flatchain)[np.argmax(sampler.flatlnprobability)]
