from typing import List

import numpy as np
import astropy.units as u

# TODO: Remove data handler at some point if possible
from ..data_processing.data_prep import DataHandler


def make_inital_guess_from_priors(priors: List[float]) -> List[float]:
    """Initialises a random float/list via a uniform distribution from the
    bounds provided

    Parameters
    -----------
    priors: IterNamespace
        Bounds list must be nested list(s) containing the bounds of the form
        form [lower_bound, upper_bound]

    Returns
    -------
    List[float]
        A list of the parameters corresponding to the priors. Also does not take the full
        priors but 1/4 from the edges, to avoid emcee problems
    """
    params = []
    for prior in priors:
        quarter_prior_distance = np.diff(prior)/4
        lower_bound, upper_bounds = prior[0]+quarter_prior_distance,\
            prior[1]-quarter_prior_distance
        param = np.random.uniform(lower_bound, upper_bounds)[0]
        params.append(param)
    return np.array(params)


def lnlike(theta: np.ndarray, data: DataHandler) -> float:
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
    lnf = theta[-1]
    total_flux_mod, corr_flux_mod, cphases_mod = calculate_model(theta[:-1], data)

    if data.fit_total_flux:
        total_flux_chi_sq = chi_sq(data.total_fluxes,
                                   data.total_fluxes_error,
                                   total_flux_mod, lnf)
    else:
        total_flux_chi_sq= 0

    corr_flux_chi_sq = chi_sq(data.corr_fluxes,
                              data.corr_fluxes_error,
                              corr_flux_mod, lnf)
    if data.fit_cphases:
        cphases_chi_sq = chi_sq(data.cphases,
                                data.cphases_error,
                                cphases_mod, lnf)
    else:
        cphases_chi_sq = 0

    return np.array(total_flux_chi_sq+corr_flux_chi_sq+cphases_chi_sq)


def lnprior(theta: np.ndarray, priors: List[List[float]]) -> float:
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
    for i, o in enumerate(priors):
        if not (o[0] < theta[i] < o[1]):
            return -np.inf
    return 0.


def lnprob(theta: np.ndarray, data: DataHandler) -> np.ndarray:
    """This function runs the lnprior and checks if it returned -np.inf, and
    returns if it does. If not, (all priors are good) it returns the inlike for
    that model (convention is lnprior + lnlike)

    Parameters
    ----------
    theta: List
        A vector that contains all the parameters of the model

    Returns
    -------
    float
        The minimisation value or -np.inf if it fails
    """
    return lnlike(theta, data) if np.isfinite(lnprior(theta, data.priors)) else -np.inf


def chi_sq(real_data: u.Quantity, data_error: u.Quantity,
           data_model: u.Quantity, lnf: float) -> float:
    """The chi square minimisation

    Parameters
    ----------
    real_data: astropy.units.Quantity
    data_error: astropy.units.Quantity
    data_model: astropy.units.Quantity
    lnf: float, optional

    Returns
    -------
    float
    """
    inv_sigma_squared = 1./np.sum(data_error.value**2+\
                                  data_model.value**2*np.exp(2*lnf))
    return -0.5*np.sum((real_data.value-data_model.value)**2*inv_sigma_squared\
                       - np.log(inv_sigma_squared))


if __name__ == "__main__":
    ...
