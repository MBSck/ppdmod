from multiprocessing import Pool
from typing import Tuple, Optional, List, Dict

import astropy.units as u
import emcee
import numpy as np

from .custom_components import assemble_components
from .fft import interpolate_coordinates
from .model import Model
from .parameter import Parameter
from .options import OPTIONS


# TODO: Check if the order is preserved here.
def set_theta_from_params(
        components_and_params: List[List[Dict]],
        shared_params: Optional[Dict[str, Parameter]] = None) -> np.ndarray:
    """Sets the theta vector from the parameters."""
    theta = []
    for (_, params) in components_and_params:
        if not params:
            continue
        theta.extend([parameter.value for parameter in params.values()])
    theta.extend([parameter.value for parameter in shared_params.values()])
    return np.array(theta)


def set_params_from_theta(theta: np.ndarray) -> float:
    """Sets the parameters from the theta vector."""
    components_and_params = OPTIONS["model.components_and_params"]
    shared_params = OPTIONS["model.shared_params"]

    new_shared_params = {}
    if shared_params is not None:
        for key, param in zip(shared_params.keys(),
                              theta[-len(shared_params):]):
            new_shared_params[key] = param

    lower, upper = None, None
    new_components_and_params, lower = [], None
    for (component, params) in components_and_params:
        if component == components_and_params[-1][0]:
            upper = -len(shared_params) if shared_params is not None else None
        new_components_and_params.append(
            [component, dict(zip(params.keys(), theta[lower:upper]))])
        lower = lower + len(params) if lower is not None else len(params)
    return new_components_and_params, new_shared_params


def init_randomly(nwalkers: float) -> np.ndarray:
    """initialises a random numpy.ndarray from the parameter's limits.

    parameters
    -----------
    components_and_params : list of list of dict

    Returns
    -------
    theta : numpy.ndarray
    """
    params = []
    for (_, param) in OPTIONS["model.components_and_params"]:
        params.extend(param.values())
    params.extend(OPTIONS["model.shared_params"].values())
    return np.array([[np.random.uniform(param.min, param.max)
                      for param in params] for _ in range(nwalkers)])


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


def calculate_observables(fourier_transform: np.ndarray,
                          vis_ucoord: np.ndarray,
                          vis_vcoord: np.ndarray,
                          cphase_ucoord: np.ndarray,
                          cphase_vcoord: np.ndarray,
                          pixel_size: u.mas,
                          wavelength: float) -> Tuple:
    """Calculates the model's observables.

    Parameters
    ----------
    fourier_transform : numpy.ndarray
        The fourier transform of the model.
    vis_ucoord : numpy.ndarray
        The u coordinate of the visibilities.
    vis_vcoord : numpy.ndarray
        The v coordinate of the visibilities.
    cphase_ucoord : numpy.ndarray
        The u coordinate of the closure phases.
    cphase_vcoord : numpy.ndarray
        The v coordinate of the closure phases.
    wavelength : float
        The wavelength of the model.

    Returns
    -------
    total_flux : float
        The total flux.
    corr_flux : float
        The correlated flux.
    cphase : float
        The closure phase.
    """
    total_flux, corr_flux, cphase = None, None, None
    if "flux" in OPTIONS["fit.data"]:
        centre = fourier_transform.shape[0]//2
        total_flux = np.abs(fourier_transform[centre, centre])

    if "vis" in OPTIONS["fit.data"]:
        interpolated_corr_flux = interpolate_coordinates(
            fourier_transform, fourier_transform.shape[0],
            pixel_size, vis_ucoord, vis_vcoord, wavelength)
        corr_flux = np.abs(interpolated_corr_flux)

    if "t3phi" in OPTIONS["fit.data"]:
        interpolated_cphase = interpolate_coordinates(
            fourier_transform, fourier_transform.shape[0],
            pixel_size, cphase_ucoord, cphase_vcoord, wavelength)
        cphase = np.angle(
            np.product(interpolated_cphase, axis=1), deg=True)
    return total_flux, corr_flux, cphase


def calculate_observables_chi_sq(
        total_flux: Dict[str, float],
        total_flux_err: Dict[str, float],
        total_flux_model: np.ndarray,
        corr_flux: Dict[str, float],
        corr_flux_err: Dict[str, float],
        corr_flux_model: np.ndarray,
        cphase: Dict[str, float],
        cphase_err: Dict[str, float],
        cphase_model: np.ndarray) -> float:
    """Calculates the model's observables.

    Parameters
    ----------
    total_flux : dict
        The total flux.
    total_flux_err : dict
        The total flux's error.
    total_flux_model : numpy.ndarray
        The model's total flux.
    corr_flux : dict
        The correlated flux.
    corr_flux_err : dict
        The correlated flux's error.
    corr_flux_model : numpy.ndarray
        The model's correlated flux.
    cphase : dict
        The closure phase.
    cphase_err : dict
        The closure phase's error.
    cphase_model : numpy.ndarray
        The model's closure phase.

    Returns
    -------
    total_chi_sq : float
        The total chi square.
    """
    total_chi_sq = 0
    if "flux" in OPTIONS["fit.data"]:
        total_chi_sq += chi_sq(total_flux, total_flux_err, total_flux_model)\
            * OPTIONS["fit.chi2.weight.total_flux"]

    if "vis" in OPTIONS["fit.data"]:
        total_chi_sq += chi_sq(corr_flux, corr_flux_err, corr_flux_model)\
            * OPTIONS["fit.chi2.weight.corr_flux"]

    if "t3phi" in OPTIONS["fit.data"]:
        total_chi_sq += chi_sq(cphase, cphase_err, cphase_model)\
            * OPTIONS["fit.chi2.weight.cphase"]
    return float(total_chi_sq)


def lnprior(components_and_params: List[List[Dict]],
            shared_params: Optional[Dict[str, float]] = None) -> float:
    """Checks if the priors are in bounds.

    Parameters
    ----------
    components_and_params : list of list of dict
        The components and parameters.
    shared_params : dict, optional
        The shared parameters.

    Returns
    -------
    float
        The log of the prior.
    """
    if shared_params is not None:
        for value, param in zip(shared_params.values(),
                                OPTIONS["model.shared_params"].values()):
            if param.free:
                if not param.min < value < param.max:
                    return -np.inf
    for (_, values), (_, params) in zip(
            components_and_params, OPTIONS["model.components_and_params"]):
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
        The parameters that ought to be fitted.

    Returns
    -------
    float
        The log of the probability.
    """
    parameters, shared_params = set_params_from_theta(theta)

    lnp = lnprior(parameters, shared_params)
    if np.isinf(lnp):
        return -np.inf

    components = assemble_components(parameters, shared_params)
    model = Model(components)

    fourier_transforms = {}
    for wavelength in OPTIONS["fit.wavelengths"]:
        fourier_transforms[str(wavelength.value)] =\
            model.calculate_complex_visibility(wavelength)

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
        readout = OPTIONS["data.readouts"][index]
        for wavelength in OPTIONS["fit.wavelengths"]:
            wavelength_str = str(wavelength.value)
            if wavelength_str not in corr_flux:
                continue
            fourier_transform = fourier_transforms[wavelength_str]
            total_flux_model, corr_flux_model, cphase_model =\
                calculate_observables(
                    fourier_transform,
                    readout.ucoord, readout.vcoord,
                    readout.u123coord, readout.v123coord,
                    components[-1].params["pixel_size"](),
                    wavelength)

            total_chi_sq += calculate_observables_chi_sq(
                total_flux[wavelength_str],
                total_flux_err[wavelength_str], total_flux_model,
                corr_flux[wavelength_str],
                corr_flux_err[wavelength_str], corr_flux_model,
                cphase[wavelength_str], cphase_err[wavelength_str],
                cphase_model)
    return total_chi_sq


def run_mcmc(nwalkers: int,
             nsteps: Optional[int] = 100,
             nsteps_burnin: Optional[int] = 0,
             ncores: Optional[int] = 6) -> np.ndarray:
    """Runs the emcee Hastings Metropolitan sampler.

    The EnsambleSampler recieves the parameters and the args are passed to
    the 'log_prob()' method (an addtional parameter 'a' can be used to
    determine the stepsize, defaults to None).

    Parameters
    ----------
    nwalkers : int, optional
    nsteps : int, optional
    discard : int, optional
    ncores : int, optional
    save_path: pathlib.Path, optional

    Returns
    -------
    sampler : numpy.ndarray

    Notes
    -----
    Burn-in should be the same as later discarded.
    """
    theta = init_randomly(nwalkers)
    ndim = theta.shape[1]
    print(f"Executing MCMC with {ncores} cores.")
    print(f"{' ':-^50}")
    with Pool(processes=ncores) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                        threads=ncores, pool=pool)
        if nsteps_burnin > 0:
            print("Running burn-in...")
            sampler.run_mcmc(theta, nsteps_burnin, progress=True)
            print("Running production...")
        sampler.reset()
        sampler.run_mcmc(theta, nsteps, progress=True)
    return sampler


def get_best_fit(sampler: emcee.EnsembleSampler,
                 discard: Optional[int] = 0) -> np.ndarray:
    """Gets the best fit from the emcee sampler."""
    samples = sampler.get_chain(flat=True, discard=discard)
    probability = sampler.get_log_prob(flat=True, discard=discard)
    return samples[np.argmax(probability)]
