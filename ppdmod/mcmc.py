from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Optional, Dict

import astropy.units as u
import emcee
import numpy as np

from .custom_components import Star, AsymmetricSDGreyBodyContinuum,\
    assemble_components
from .fft import interpolate_coordinates
from .model import Model
from .parameter import Parameter
from .options import OPTIONS
# from .utils import execution_time


def set_theta_from_params(
        component_and_params: Dict[str, Dict],
        shared_params: Optional[Dict[str, Parameter]] = None) -> np.ndarray:
    """Sets the theta vector from the parameters."""
    theta = []
    for component in component_and_params.values():
        theta.extend([parameter.value for parameter in component.values()])
    theta.extend([parameter.value for parameter in shared_params.values()])
    return np.array(theta)


def set_params_from_theta(
        theta: np.ndarray,
        components_and_params: Dict[str, Dict],
        shared_params: Optional[Dict[str, Parameter]] = None) -> float:
    """Sets the parameters from the theta vector."""
    new_shared_params = {}
    if shared_params is not None:
        for key, value in zip(shared_params.keys(),
                              theta[-len(shared_params):]):
            new_shared_params[key] = value

    lower, upper = None, None
    new_components_and_params, lower = {}, None
    for component, value in components_and_params.items():
        if component == list(components_and_params.keys())[-1]:
            upper = -len(shared_params) if shared_params is not None else None
        new_components_and_params[component] =\
            dict(zip(value.keys(), theta[lower:upper]))
        lower = lower + len(value) if lower is not None else len(value)
    return new_components_and_params, new_shared_params


def init_randomly(free_parameters: Dict[str, Parameter],
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


def chi_sq(data: u.quantity, error: u.quantity,
           model_data: u.quantity, lnf: Optional[float] = None) -> float:
    """the chi square minimisation.

    Parameters
    ----------
    data : numpy.ndarray
        The real data.
    error : numpy.ndarray
        The real data's error.
    model_data : numpy.ndarray
        The model data.
    lnf : float, optional
        The error correction term for the real data.

    Returns
    -------
    chi_sq : float
    """
    if lnf is None:
        inv_sigma_squared = 1./np.sum(error**2)
    else:
        inv_sigma_squared = 1./np.sum(
            error**2 + model_data**2*np.exp(2*lnf))
    return -0.5*np.sum((data-model_data)**2*inv_sigma_squared)


def lnprior(param_values: Dict[str, float],
            shared_param_values: Optional[Dict[str, float]] = None) -> float:
    """Checks if the priors are in bounds."""
    if shared_param_values is not None:
        for value, param in zip(shared_param_values.values(),
                                OPTIONS["model.shared_params"].values()):
            if not param.min < value < param.max:
                return -np.inf
    for values, params in zip(param_values.values(),
                              OPTIONS["model.components_and_params"].values()):
        for value, param in zip(values.values(), params.values()):
            if param.free:
                if not param.min < value < param.max:
                    return -np.inf
    return 0


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
    parameters, shared_params = set_params_from_theta(
        theta,
        OPTIONS["model.components_and_params"],
        OPTIONS["model.shared_params"])

    if np.isinf(lnprior(parameters, shared_params)):
        return -np.inf

    components = assemble_components(parameters, shared_params)
    model = Model(components)

    fourier_transforms = {}
    for wavelength in OPTIONS["fit.wavelengths"]:
        fourier_transform = model.calculate_complex_visibility(wavelength)
        fourier_transforms[str(wavelength)] = fourier_transform

    total_fluxes, total_fluxes_err =\
        OPTIONS["data.total_flux"], OPTIONS["data.total_flux_error"]
    corr_fluxes, corr_fluxes_err =\
        OPTIONS["data.correlated_flux"], OPTIONS["data.correlated_flux_error"]
    cphases, cphases_err =\
        OPTIONS["data.closure_phase"], OPTIONS["data.closure_phase_error"]

    total_chi_sq = 0
    for index, (total_flux, total_flux_err, corr_flux,
                corr_flux_err, cphase, cphase_err)\
            in enumerate(
                zip(total_fluxes, total_fluxes_err, corr_fluxes,
                    corr_fluxes_err, cphases, cphases_err)):

        for wavelength in OPTIONS["fit.wavelengths"]:
            if str(wavelength) in fourier_transforms:
                fourier_transform = fourier_transforms[str(wavelength)]

                if "flux" in OPTIONS["fit.data"]:
                    centre = fourier_transform.shape[0]//2
                    total_flux_model = fourier_transform[centre, centre]
                    total_chi_sq += chi_sq(total_flux[str(wavelength)],
                                           total_flux_err[str(wavelength)],
                                           total_flux_model)\
                        * OPTIONS["fit.chi2.weight.total_flux"]

                if "vis" in OPTIONS["fit.data"]:
                    interpolated_corr_flux = interpolate_coordinates(
                        fourier_transform,
                        fourier_transform.shape[0],
                        components.params["pixel_size"](),
                        OPTIONS["data.readouts"][index].ucoord,
                        OPTIONS["data.readouts"][index].vcoord,
                        wavelength)
                    corr_flux_model = np.abs(interpolated_corr_flux)
                    total_chi_sq += chi_sq(corr_flux[str(wavelength)],
                                           corr_flux_err[str(wavelength)],
                                           corr_flux_model)\
                        * OPTIONS["fit.chi2.weight.corr_flux"]

                if "t3phi" in OPTIONS["fit.data"]:
                    interpolated_cphase = interpolate_coordinates(
                        fourier_transform,
                        fourier_transform.shape[0],
                        components.params["pixel_size"](),
                        OPTIONS["data.readouts"][index].u123coord,
                        OPTIONS["data.readouts"][index].v123coord,
                        wavelength)
                    cphase_model = np.angle(
                        np.product(interpolated_cphase, axis=1), deg=True)
                    total_chi_sq += chi_sq(cphase[str(wavelength)],
                                           cphase_err[str(wavelength)],
                                           cphase_model)\
                        * OPTIONS["fit.chi2.weight.cphase"]
            else:
                continue
    return np.array(total_chi_sq)


def run_mcmc(nwalkers: int,
             nsteps: Optional[int] = 100,
             discard: Optional[int] = 10,
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
    initial = init_randomly(OPTIONS["model.params"], nwalkers)
    # with Pool() as pool:
    print(f"Executing MCMC with {cpu_count()} cores.")
    print("--------------------------------------------------------------")
    sampler = emcee.EnsembleSampler(nwalkers, len(OPTIONS["model.params"]),
                                    lnprob, pool=None)
    sampler.run_mcmc(initial, nsteps, progress=True)
    return sampler


def get_best_fit(sampler: emcee.EnsembleSampler) -> np.ndarray:
    """Gets the best fit from the emcee sampler."""
    return (sampler.flatchain)[np.argmax(sampler.flatlnprobability)]
