import numpy as np

from astropy.units import Quantity

from .combined_model import CombinedModel
from .fourier import FastFourierTransform
from .data_prep import DataHandler


def calculate_model(theta: np.ndarray, data: DataHandler):
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
        data.fourier = FastFourierTransform(image, wavelength,
                                            data.pixel_scaling, data.zero_padding_order)
        corr_flux_data, cphases_data = [], []
        corr_flux, cphases = data.fourier.get_uv2fft2(data.uv_coords, data.uv_coords_cphase)
        corr_flux_data.extend(corr_flux)
        cphases_data.extend(cphases)
        total_flux_mod_chromatic.append(total_flux_data)
        corr_flux_mod_chromatic.append(corr_flux_data)
        cphases_mod_chromatic_data.append(cphases_data)
    return total_flux_mod_chromatic, corr_flux_mod_chromatic, cphases_mod_chromatic_data


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
    total_flux_mod, corr_flux_mod, cphases_mod = calculate_model(theta, data)
    total_flux_chi_sq = chi_sq(data.total_fluxes.value,
                               data.total_fluxes_sigma_squared.value, total_flux_mod)
    corr_flux_chi_sq = chi_sq(data.corr_fluxes.value,
                              data.corr_fluxes_sigma_squared.value, corr_flux_mod)
    cphases_chi_sq = chi_sq(data.cphases.value,
                            data.cphases_sigma_squared.value, cphases_mod)
    return np.array(-0.5*(total_flux_chi_sq +\
                          corr_flux_chi_sq + cphases_chi_sq), dtype=float)

def lnprior(theta: np.ndarray, data: DataHandler) -> float:
    """Checks if all variables are within their priors (as well as
    determining them setting the same).

    If all priors are satisfied it needs to return '0.0' and if not '-np.inf'
    This function checks for an unspecified amount of flat priors. If upper
    bound is 'None' then no upper bound is given

    Parameters
    ----------
    theta: np.ndarray
        A list of all the parameters that ought to be fitted
    priors: List
        A list containing all the prior's bounds

    Returns
    -------
    float
        Return-code 0.0 for within bounds and -np.inf for out of bound
        priors
    """
    check_conditons = []

    for i, o in enumerate(data.priors):
        if o[1] is None:
            if o[0] < theta[i]:
                check_conditons.append(True)
            else:
                check_conditons.append(False)
        else:
            if o[0] < theta[i] < o[1]:
                check_conditons.append(True)
            else:
                check_conditons.append(False)

    return 0.0 if all(check_conditons) else -np.inf

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
    lp = lnprior(theta, data)

    if not np.isfinite(lp):
        return -np.inf

    return lp + lnlike(theta, data)

def chi_sq(real_data: Quantity, sigma_sq: Quantity, model_data: Quantity) -> float:
    """The chi square minimisation"""
    return np.sum(np.log(2*np.pi*sigma_sq) + (real_data-model_data)**2/sigma_sq)


if __name__ == "__main__":
    ...

