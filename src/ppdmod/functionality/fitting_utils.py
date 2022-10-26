import numpy as np

from typing import Optional, List
from astropy.units import Quantity

from .data_prep import DataHandler
from .combined_model import CombinedModel
from .fourier import FastFourierTransform


def calculate_model(theta: np.ndarray, data: DataHandler,
                    rfourier: Optional[bool] = False):
    data.reformat_theta_to_components(theta)
    model = CombinedModel(data.fixed_params, data.disc_params,
                          data.wavelengths, data.geometric_params,
                          data.modulation_params)
    model.tau = data.tau_initial
    for component in data.model_components:
        model.add_component(component)

    total_flux_mod_chromatic, corr_flux_mod_chromatic, cphases_mod_chromatic_data =\
        [], [], []
    for i, wavelength in enumerate(data.wavelengths):
        image = model.eval_flux(wavelength)
        total_flux_data = []
        for _ in data.total_fluxes[i]:
             total_flux_data.append(model.eval_total_flux(wavelength).value)
        fourier = FastFourierTransform(image, wavelength,
                                            data.pixel_scaling, data.zero_padding_order)
        corr_flux_data, cphases_data = [], []
        corr_flux, cphases = fourier.get_uv2fft2(data.uv_coords, data.uv_coords_cphase)
        corr_flux_data.extend(corr_flux)
        cphases_data.extend(cphases)
        total_flux_mod_chromatic.append(total_flux_data)
        corr_flux_mod_chromatic.append(corr_flux_data)
        cphases_mod_chromatic_data.append(cphases_data)
    model_data = [total_flux_mod_chromatic,
                  corr_flux_mod_chromatic, cphases_mod_chromatic_data]
    if rfourier:
        model_data.insert(len(model_data), fourier)
    return model_data


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
    # total_flux_chi_sq = chi_sq(data.total_fluxes.value,
                               # data.total_fluxes_error.value,
                               # total_flux_mod, lnf)
    corr_flux_chi_sq = chi_sq(data.corr_fluxes.value,
                              data.corr_fluxes_error.value,
                              corr_flux_mod, lnf)
    # cphases_chi_sq = chi_sq(data.cphases.value,
                            # data.cphases_error.value,
                            # cphases_mod, lnf)
    return np.array(corr_flux_chi_sq)

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

def chi_sq(real_data: Quantity, data_error: Quantity,
           data_model: Quantity, lnf: float) -> float:
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
    inv_sigma_squared = 1./(data_error**2+data_model**2*np.exp(2*lnf))
    return -0.5*np.sum((real_data-data_model)**2*inv_sigma_squared\
                       - np.log(inv_sigma_squared))


if __name__ == "__main__":
    ...

