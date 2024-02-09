from multiprocessing import Pool
from typing import Optional, List, Dict

import astropy.units as u
import emcee
import numpy as np
from scipy.stats import gaussian_kde

from .custom_components import assemble_components
from .component import Component
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
           model_data: u.quantity,
           method: Optional[str] = "linear",
           lnf: Optional[float] = None) -> float:
    """the chi square minimisation.

    Parameters
    ----------
    data : numpy.ndarray
        The real data.
    error : numpy.ndarray
        The real data's error.
    model_data : numpy.ndarray
        The model data.
    method : str, optional
        The method of comparison, either "linear" or "exponential".
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
    if method == "linear":
        return -0.5*np.sum((data-model_data)**2*inv_sigma_squared + np.log(1/inv_sigma_squared))
    diff = np.angle(np.exp((data-model_data)*u.deg.to(u.rad)*1j), deg=False)
    return -0.5*np.sum(diff**2*inv_sigma_squared + np.log(1/inv_sigma_squared))


def calculate_observables(components: List[Component], wavelength: u.um,
                          ucoord: np.ndarray, vcoord: np.ndarray,
                          u123coord: np.ndarray, v123coord: np.ndarray):
    """Calculates the observables from the model."""
    stellar_flux = components[0].calculate_stellar_flux(wavelength)

    flux_model, corr_flux_model, cphase_model = None, None, None
    for component in components[1:]:
        tmp_flux = component.calculate_total_flux(wavelength)
        tmp_corr_flux = component.calculate_visibility(
                ucoord, vcoord, wavelength)
        tmp_cphase = component.calculate_closure_phase(
                u123coord, v123coord, wavelength)

        if flux_model is None:
            flux_model = tmp_flux
            corr_flux_model = tmp_corr_flux
            cphase_model = tmp_cphase
        else:
            flux_model += tmp_flux
            corr_flux_model += tmp_corr_flux
            cphase_model += tmp_cphase
    flux_model += stellar_flux
    corr_flux_model += stellar_flux
    return flux_model, corr_flux_model, cphase_model


def calculate_observables_chi_sq(
        flux: np.ndarray,
        flux_err: np.ndarray,
        flux_model: np.ndarray,
        vis: np.ndarray,
        vis_err: np.ndarray,
        corr_flux_model: np.ndarray,
        cphase: np.ndarray,
        cphase_err: np.ndarray,
        cphase_model: np.ndarray) -> float:
    """Calculates the model's observables.

    Parameters
    ----------
    flux : numpy.ndarray
        The total flux.
    flux_err : numpy.ndarray
        The total flux's error.
    flux_model : numpy.ndarray
        The model's total flux.
    vis : numpy.ndarray
        Either the correlated fluxes or the visibilities.
    corr_flux_err : numpy.ndarray
        Either the error of the correlated fluxes or the
        error of the visibilities.
    corr_flux_model : numpy.ndarray
        The model's correlated flux.
    cphase : numpy.ndarray
        The closure phase.
    cphase_err : numpy.ndarray
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
        total_chi_sq += chi_sq(flux, flux_err, flux_model)\
            * OPTIONS["fit.chi2.weight.flux"]

    if "vis" in OPTIONS["fit.data"] or "vis2" in OPTIONS["fit.data"]:
        if "vis" in OPTIONS["fit.data"]:
            vis_model = corr_flux_model
        else:
            vis_model = corr_flux_model/flux_model
        total_chi_sq += chi_sq(vis, vis_err, vis_model)\
            * OPTIONS["fit.chi2.weight.corr_flux"]

    if "t3phi" in OPTIONS["fit.data"]:
        total_chi_sq += chi_sq(cphase, cphase_err,
                               cphase_model, method="exponential")\
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


def lnprob(theta: np.ndarray) -> float:
    """Takes theta vector and the x, y and the yerr of the theta.
    Returns a number corresponding to how good of a fit the model is to your
    data for a given set of parameters, weighted by the data points.

    This is the analytical 1D implementation.

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

    # HACK: This is to include innermost radius for rn.
    innermost_radius = components[1].params["rin"]
    for component in components:
        component.params["rin0"] = innermost_radius

    fluxes, fluxes_err =\
        OPTIONS["data.flux"], OPTIONS["data.flux_err"]

    if "vis" in OPTIONS["fit.data"]:
        vis, vis_err =\
            OPTIONS["data.corr_flux"], OPTIONS["data.corr_flux_err"]
        ucoord, vcoord =\
            OPTIONS["data.corr_flux.ucoord"], OPTIONS["data.corr_flux.vcoord"]
    else:
        vis, vis_err =\
            OPTIONS["data.vis"], OPTIONS["data.vis_err"]
        ucoord, vcoord =\
            OPTIONS["data.vis.ucoord"], OPTIONS["data.vis.vcoord"]

    cphases, cphases_err =\
        OPTIONS["data.cphase"], OPTIONS["data.cphase_err"]
    u123coord = OPTIONS["data.cphase.u123coord"]
    v123coord = OPTIONS["data.cphase.v123coord"]

    total_chi_sq = 0
    for index, wavelength in enumerate(OPTIONS["fit.wavelengths"]):
        flux_model, corr_flux_model, cphase_model = calculate_observables(
                components, wavelength, ucoord[index],
                vcoord[index], u123coord[index], v123coord[index])
        total_chi_sq += calculate_observables_chi_sq(
                fluxes[index], fluxes_err[index], flux_model,
                vis[index], vis_err[index], corr_flux_model,
                cphases[index], cphases_err[index], cphase_model)
    return total_chi_sq


def run_mcmc(nwalkers: int,
             nsteps: Optional[int] = 100,
             nburnin: Optional[int] = 0,
             ncores: Optional[int] = 6,
             debug: Optional[bool] = False) -> np.ndarray:
    """Runs the emcee Hastings Metropolitan sampler.

    The EnsambleSampler recieves the parameters and the args are passed to
    the 'log_prob()' method (an addtional parameter 'a' can be used to
    determine the stepsize, defaults to None).

    Parameters
    ----------
    nwalkers : int, optional
    theta : numpy.ndarray
    nsteps : int, optional
    discard : int, optional
    ncores : int, optional
    save_path: pathlib.Path, optional

    Returns
    -------
    sampler : numpy.ndarray
    """
    theta = init_randomly(nwalkers)
    print(f"Executing MCMC with {ncores} cores.")
    print(f"{'':-^50}")
    if debug:
        sampler = emcee.EnsembleSampler(
            nwalkers, theta.shape[1], lnprob, pool=None)
        if nburnin is not None:
            print("Running burn-in...")
            sampler.run_mcmc(theta, nburnin, progress=True)
            print("Running production...")
        sampler.reset()
        sampler.run_mcmc(theta, nsteps, progress=True)
    else:
        with Pool(processes=ncores) as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers, theta.shape[1], lnprob, pool=pool)
            if nburnin is not None:
                print("Running burn-in...")
                sampler.run_mcmc(theta, nburnin, progress=True)
                print("Running production...")
            sampler.reset()
            sampler.run_mcmc(theta, nsteps, progress=True)
    return sampler


def run_dynesty(debug: Optional[bool] = False) -> np.ndarray:
    """Runs the dynesty sampler."""
    sampler = None
    if debug:
        ...
    else:
        ...
    return sampler


def run_fit(**kwargs) -> np.ndarray:
    """Runs the fit using either standard or nested sampling.

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
    """
    if OPTIONS["fit.method"] == "emcee":
        return run_mcmc(**kwargs)
    return run_dynesty(**kwargs)


def get_best_fit(sampler: emcee.EnsembleSampler,
                 discard: Optional[int] = 0,
                 method: Optional[str] = "gaussian") -> np.ndarray:
    """Gets the best fit from the emcee sampler."""
    samples = sampler.get_chain(flat=True, discard=discard)
    if method == "gaussian":
        kde = gaussian_kde(samples.T)
        probability = kde.pdf(samples.T)
    probability = sampler.get_log_prob(flat=True, discard=discard)
    return samples[np.argmax(probability)]
