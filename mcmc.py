from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Optional, Callable, Dict, List

import emcee
import numpy as np

from parameter import Parameter


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
def lnlike(theta: np.ndarray) -> float:
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
    # if data.fit_total_flux:
    #     total_flux_chi_sq = chi_sq(data.total_fluxes,
    #                                data.total_fluxes_error,
    #                                total_flux_mod, lnf)
    # else:
    #     total_flux_chi_sq= 0
    #
    # corr_flux_chi_sq = chi_sq(data.corr_fluxes,
    #                           data.corr_fluxes_error,
    #                           corr_flux_mod, lnf)
    # if data.fit_cphases:
    #     cphases_chi_sq = chi_sq(data.cphases,
    #                             data.cphases_error,
    #                             cphases_mod, lnf)
    # else:
    #     cphases_chi_sq = 0
    # return np.array(total_flux_chi_sq+corr_flux_chi_sq+cphases_chi_sq)


def lnprior(parameters: np.ndarray, priors: List[List[float]]) -> float:
    """Checks if all variables are within their priors (as well as
    determining them setting the same).

    If all priors are satisfied it needs to return '0.0' and if not '-np.inf'
    This function checks for an unspecified amount of flat priors. If upper
    bound is 'None' then no upper bound is given

    Parameters
    ----------
    theta: np.ndarray
        A list of all the parameters that ought to be fitted
    priors: List[List[float]]
        A list containing all the priors' bounds

    Returns
    -------
    float
        Return-code 0.0 for within bounds and -np.inf for out of bound priors
    """
    for parameter in parameters:
        if parameter.free:
            if not parameter.limits[0] < parameter.value < parameter.limits[1]:
                return -np.inf
    return 0.

def lnprob(parameters: Dict[Parameter]) -> np.ndarray:
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
    return lnlike(theta, data) if np.isfinite(lnprior(parameters)) else -np.inf


def initiate_random(parameters: Dict[Parameter]) -> np.array:
    """initialises a random numpy.ndarray from the parameter's limits.

    parameters
    -----------
    parameters : dict of Parameter

    Returns
    -------
    numpy.ndarray
    """
    theta = [parameter.value for parameter in parameters]
    nw = self.params['nwalkers'].value
    initialParams = np.ndarray([nw, self.nfree])

    for iparam, parami in enumerate(self.freeParams.values()):
        initialParams[:, iparam] = np.random.random(
            self.params['nwalkers'].value)*(parami.max-parami.min)+parami.min

    return initialParams


def generate_valid_guess(initial: List, priors: List,
                         nwalkers: int, frac: float) -> np.ndarray:
    """Generates a valid guess that is in the bounds of the priors for the
    start of the MCMC-fitting

    Parameters
    ----------
    inital: List
        The inital guess
    priors: List
        The priors that constrain the guess
    nwalkers: int
        The number of walkers to be initialised for

    Returns
    -------
    p0: np.ndarray
        A valid guess for the number of walkers according to the number of
        dimensions
    """
    proposal = np.array(initial)
    prior_dynamic = np.array([np.ptp(i) for i in priors])
    dyn = 1/prior_dynamic

    guess_lst = []
    for _ in range(nwalkers):
        guess = proposal + frac*dyn*np.random.normal(proposal, dyn)
        guess_lst.append(guess)

    return np.array(guess_lst, dtype=float)


def run_mcmc(parameters: Parameter,
             nwalkers: int,
             nsteps: int,
             discard: int,
             wavelength: float,
             frac: Optional[float] = 1e-4,
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
    lnprob: Callable
    plot_wl: float
    frac: float, optional
    save_path: pathlib.Path, optional
    """
    # p0 = generate_valid_guess(initial, nwalkers, frac)
    with Pool() as pool:
        print(f"Executing MCMC with {cpu_count()} cores.")
        print("--------------------------------------------------------------")
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool)
        # pos, prob, state = sampler.run_mcmc(p0, nsteps, progress=True)
    theta_max = (sampler.flatchain)[np.argmax(sampler.flatlnprobability)]
