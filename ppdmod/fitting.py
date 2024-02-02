from multiprocessing import Pool
from types import SimpleNamespace
from typing import Optional, List, Dict, Tuple, Union

import astropy.units as u
import emcee
import numpy as np
# from dynesty import NestedSampler
from scipy.stats import gaussian_kde

from .custom_components import assemble_components
from .component import Component
from .parameter import Parameter
from .options import OPTIONS


def get_priors() -> np.ndarray:
    """Gets the priors from the model parameters."""
    priors = []
    for _, params in OPTIONS.model.components_and_params:
        for param in params.values():
            if param.free:
                priors.append([param.min, param.max])

    if OPTIONS.model.shared_params is not None:
        for param in OPTIONS.model.shared_params.values():
            if param.free:
                priors.append([param.min, param.max])
    return np.array(priors)


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


def set_params_from_theta(
        theta: np.ndarray
        ) -> Tuple[List[Dict[str, Component]], Dict[str, Parameter]]:
    """Sets the parameters from the theta vector."""
    components_and_params = OPTIONS.model.components_and_params
    shared_params = OPTIONS.model.shared_params

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


def init_randomly(nwalkers: Optional[int] = None) -> np.ndarray:
    """initialises a random numpy.ndarray from the parameter's limits.

    parameters
    -----------
    components_and_params : list of list of dict

    Returns
    -------
    theta : numpy.ndarray
    """
    params = []
    for (_, param) in OPTIONS.model.components_and_params:
        params.extend(param.values())
    params.extend(OPTIONS.model.shared_params.values())
    if nwalkers is None:
        return np.array([np.random.uniform(param.min, param.max)
                         for param in params])
    return np.array([[np.random.uniform(param.min, param.max)
                      for param in params] for _ in range(nwalkers)])


def calculate_chi_sq(data: u.quantity, error: u.quantity,
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
        return -0.5*np.sum((data-model_data)**2*inv_sigma_squared
                           + np.log(1/inv_sigma_squared))

    diff = np.angle(np.exp((data-model_data)*u.deg.to(u.rad)*1j), deg=False)
    return -0.5*np.sum(diff**2*inv_sigma_squared + np.log(1/inv_sigma_squared))


def calculate_observables(components: List[Component]):
    """Calculates the observables from the model."""
    corr_flux = "vis2" not in OPTIONS.fit.data
    vis = OPTIONS.data.vis if corr_flux else OPTIONS.data.vis2
    ucoord, vcoord = vis.ucoord, vis.vcoord
    u123coord = OPTIONS.data.t3.u123coord
    v123coord = OPTIONS.data.t3.v123coord
    wavelength = OPTIONS.fit.wavelengths

    flux_model, vis_model, t3_model = None, None, None
    if components is not None:
        stellar_flux = components[0].calculate_flux(wavelength).value
        for component in components[1:]:
            tmp_flux = component.calculate_flux(wavelength)
            tmp_vis = component.calculate_visibility(
                    ucoord, vcoord, wavelength)
            tmp_t3 = component.calculate_closure_phase(
                    u123coord, v123coord, wavelength)

            if flux_model is None:
                flux_model, vis_model, t3_model = tmp_flux, tmp_vis, tmp_t3
            else:
                flux_model += tmp_flux
                vis_model += tmp_vis
                t3_model += tmp_t3

        flux_model += stellar_flux
        vis_model += stellar_flux
        if not corr_flux:
            vis_model = vis_model/flux_model

        flux_model = np.tile(
                flux_model, (OPTIONS.data.flux.value.shape[1]))
    return flux_model, vis_model, t3_model


def calculate_observable_chi_sq(
        flux_model: np.ndarray,
        vis_model: np.ndarray,
        t3_model: np.ndarray) -> float:
    """Calculates the model's observables.

    Parameters
    ----------
    flux_model : numpy.ndarray
        The model's total flux.
    vis_model : numpy.ndarray
        Either the model's correlated fluxes or the model's
        visibilities (depends on the OPTIONS.fit.data).
    t3_model : numpy.ndarray
        The model's closure phase.

    Returns
    -------
    chi_sq : float
        The chi square.
    """
    params = {"flux": flux_model,
              "vis": vis_model, "t3": t3_model}
    chi_sq = 0.
    for key in OPTIONS.fit.data:
        data = getattr(OPTIONS.data, key)
        weight = getattr(OPTIONS.fit.weights, key)
        nan_indices = np.isnan(data.value)
        chi_sq += calculate_chi_sq(
                data.value[~nan_indices],
                data.err[~nan_indices],
                params[key][~nan_indices]) * weight
    return float(chi_sq)


# NOTE: In nested fitting priors can depend on each other
# use that in the future
# TODO: Finish this
def transform_uniform_prior(x, a, b):
    """Prior transform for uniform priors."""
    return a + (b-a)*x


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
                                OPTIONS.model.shared_params.values()):
            if param.free:
                if not param.min < value < param.max:
                    return -np.inf
    for (_, values), (_, params) in zip(
            components_and_params, OPTIONS.model.components_and_params):
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

    if OPTIONS.fit.method == "emcee":
        lnp = lnprior(parameters, shared_params)
        if np.isinf(lnp):
            return -np.inf

    components = assemble_components(parameters, shared_params)

    # HACK: This is to include innermost radius for rn.
    innermost_radius = components[1].params["rin"]
    for component in components:
        component.params["rin0"] = innermost_radius

    return calculate_observable_chi_sq(*calculate_observables(components))


def run_mcmc(nwalkers: int,
             nsteps_burnin: Optional[int] = 0,
             nsteps: Optional[int] = 100,
             ncores: Optional[int] = 6,
             debug: Optional[bool] = False,
             **kwargs) -> np.ndarray:
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
    debug : bool, optional

    Returns
    -------
    sampler : numpy.ndarray
    """
    theta = init_randomly(nwalkers)
    pool = Pool(processes=ncores) if not debug else None

    print(f"Executing MCMC.\n{'':-^50}")
    sampler = emcee.EnsembleSampler(
        nwalkers, theta.shape[1], lnprob, pool=pool)
    if nsteps_burnin is not None:
        print("Running burn-in...")
        sampler.run_mcmc(theta, nsteps_burnin, progress=True)

    sampler.reset()
    print("Running production...")
    sampler.run_mcmc(theta, nsteps, progress=True)

    if not debug:
        pool.close()
        pool.join()
    return sampler


# def run_dynesty(nlive: Optional[int] = 1000,
#                 sample: Optional[str] = "rwalk",
#                 bound: Optional[str] = "multi",
#                 ncores: Optional[int] = 6,
#                 debug: Optional[bool] = False,
#                 **kwargs) -> np.ndarray:
#     """Runs the dynesty nested sampler.

#     Parameters
#     ----------
#     ncores : int, optional
#     debug : bool, optional

#     Returns
#     -------
#     sampler : numpy.ndarray
#     """
#     ndim = init_randomly().shape[0]
#     pool = Pool(processes=ncores) if not debug else None
#     queue_size = ncores if not debug else None
#     sampler_kwargs = {"nlive": nlive, "sample": sample,
#                       "ptform_args": get_priors(),
#                       "bound": bound, "queue_size": queue_size,
#                       "pool": pool, "update_interval": ndim}

#     print(f"Executing Dynesty.\n{'':-^50}")
#     sampler = NestedSampler(lnprob, transform_uniform_prior,
#                             ndim, **sampler_kwargs)
#     sampler.run_nested(dlogz=0.01, print_progress=True)

#     if not debug:
#         pool.close()
#         pool.join()
#     return sampler


def run_fit(**kwargs) -> np.ndarray:
    """Runs the fit using either standard or nested sampling.

    Parameters
    ----------
    nwalkers : int, optional
        The number of walkers in the emcee sampler.
    nsteps : int, optional
        The number of steps in the emcee sampler.
    discard : int, optional
        The number of steps to discard in the emcee sampler.
    ncores : int, optional
        The number of cores to use in the emcee sampler.

    Returns
    -------
    sampler : numpy.ndarray
    """
    if OPTIONS.fit.method == "emcee":
        return run_mcmc(**kwargs)
    # return run_dynesty(**kwargs)


def get_best_fit(
        sampler: Union[emcee.EnsembleSampler],
        discard: Optional[int] = 0,
        distribution: Optional[str] = "default",
        method: Optional[str] = "quantile"
        ) -> Tuple[np.ndarray, np.ndarray]:
    """Gets the best fit from the emcee sampler."""
    params, uncertainties = [], []
    if OPTIONS.fit.method == "emcee":
        samples = sampler.get_chain(flat=True, discard=discard)
        if distribution == "gaussian":
            kde = gaussian_kde(samples.T)
            probability = kde.pdf(samples.T)
        else:
            probability = sampler.get_log_prob(flat=True, discard=discard)

        if method == "quantile":
            for index in range(samples.shape[1]):
                quantiles = np.percentile(samples[:, index],
                                          OPTIONS.fit.quantiles)
                params.append(quantiles[1])
                uncertainties.append(np.diff(quantiles))
            params, uncertainties = map(np.array, (params, uncertainties))
        elif method == "maximum":
            params = samples[np.argmax(probability)]
        return params, uncertainties
    else:
        ...
