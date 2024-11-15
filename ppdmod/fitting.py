from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Tuple

import astropy.units as u
import dynesty.utils as dyutils
import numpy as np
from dynesty import DynamicNestedSampler, NestedSampler

from .component import Component
from .data import get_counts_data
from .options import OPTIONS
from .parameter import Parameter
from .utils import compute_t3, compute_vis


def get_priors() -> np.ndarray:
    """Gets the priors from the model parameters."""
    priors = []
    for _, params in OPTIONS.model.components_and_params:
        for param in params.values():
            priors.append([param.min, param.max])

    if OPTIONS.model.shared_params is not None:
        for param in OPTIONS.model.shared_params.values():
            priors.append([param.min, param.max])

    return np.array(priors)


def get_labels(
    components: List[Component], shared_params: Dict[str, Parameter] | None = None
) -> np.ndarray:
    """Sets the theta vector from the components.

    Parameters
    ----------
    components : list of Component
        The components to be used in the model.
    shared_params : dict
        The shared parameters.

    Returns
    -------
    theta : numpy.ndarray
    """
    disc_models = ["TempGrad", "AsymTempGrad", "GreyBody", "AsymGreyBody"]
    labels, zone_count = [], 1
    for component in components:
        component_labels = [key for key in component.get_params(free=True)]
        if component.shortname == "Star":
            component_labels = [rf"{label}-\star" for label in component_labels]

        if component.shortname in disc_models:
            component_labels = [f"{label}-{zone_count}" for label in component_labels]
            zone_count += 1

        labels.extend(component_labels)

    if shared_params is not None:
        labels.extend([rf"{key}-\mathrm{{sh}}" for key in shared_params])

    return labels


def get_theta(
    components: List[Component], shared_params: Dict[str, Parameter] | None = None
) -> np.ndarray:
    """Sets the theta vector from the components.

    Parameters
    ----------
    components : list of Component
        The components to be used in the model.
    shared_params : dict
        The shared parameters.

    Returns
    -------
    theta : numpy.ndarray
    """
    theta = []
    for component in components:
        theta.extend(
            [param.value for param in component.get_params(free=True).values()]
        )

    if shared_params is not None:
        theta.extend([param.value for param in shared_params.values()])

    return np.array(theta)


def set_components_from_theta(theta: np.ndarray) -> List[Component]:
    """Sets the components from theta."""
    components = list(map(Component.copy, OPTIONS.model.components))
    nshared = len(OPTIONS.model.shared_params)
    theta_list, shared_params = theta[:-nshared].copy().tolist(), theta[-nshared:]
    for component in components:
        for param in component.get_params(free=True).values():
            param.value = theta_list.pop(0)

        for param_name, param in zip(OPTIONS.model.shared_params.keys(), shared_params):
            getattr(component, param_name).value = param

    return components


def compute_chi_sq(
    data: u.Quantity,
    error: u.Quantity,
    model_data: u.Quantity,
    ndim: int,
    diff_method: str = "linear",
    method: str = "logarithmic",
    lnf: float | None = None,
) -> float:
    """Computes the chi square minimisation.

    Parameters
    ----------
    data : numpy.ndarray
        The real data.
    error : numpy.ndarray
        The real data's error.
    model_data : numpy.ndarray
        The model data.
    ndim : int
        The number of (parameter) dimensions.
    diff_method : str, optional
        The method to determine the difference of the dataset,
        to the data. Either "linear" or "exponential".
        Default is "linear".
    method : str, optional
        The method used to calculate the chi square.
        Either "linear" or "logarithmic".
        Default is "logarithic".
    lnf : float, optional
        The error correction term for the real data.

    Returns
    -------
    chi_sq : float
    """
    sigma_squared = error**2
    if lnf is not None:
        sigma_squared = error**2 + model_data**2 * np.exp(2 * lnf)

    diff = data - model_data
    if diff_method != "linear":
        diff = np.exp(1j * np.deg2rad(data)) * np.exp(-1j * np.deg2rad(model_data))
        diff = np.angle(diff, deg=True)

    chi_sq = diff**2 / sigma_squared
    if method == "linear":
        return chi_sq.sum()

    lnorm = np.log(2 * np.pi) * ndim + np.log(sigma_squared)
    return -0.5 * (chi_sq + lnorm).sum()


# TODO: Write tests (ASPRO) that tests multiple components with the total flux
# TODO: Make it so that both point source and star can be used at the same time
def compute_observables(
    components: List[Component],
    wavelength: np.ndarray | None = None,
    rzero: bool = False,
    rcomponents: bool = False,
    rraw: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculates the observables from the model.

    Parameters
    ----------
    components : list of Component
        The components to be used in the model.
    wavelength : numpy.ndarray, optional
        The wavelength to be used in the model.
    rzero : bool, optional
        Whether to include the zero baseline.
    rcomponents : bool, optional
        Whether to return the individual components.
    rraw : bool, optional
        Whether to return the raw observables.
    """
    wavelength = OPTIONS.fit.wavelengths if wavelength is None else wavelength
    vis = OPTIONS.data.vis2 if "vis2" in OPTIONS.fit.data else OPTIONS.data.vis

    complex_vis_comps, complex_t3_comps = [], []
    for component in [comp for comp in components if comp.name != "Point Source"]:
        complex_vis_comps.append(
            component.compute_complex_vis(vis.ucoord, vis.vcoord, wavelength)
        )
        complex_t3_comps.append(
            component.compute_complex_vis(
                OPTIONS.data.t3.u123coord, OPTIONS.data.t3.v123coord, wavelength
            )
        )

    complex_vis_comps = np.array(complex_vis_comps)
    complex_t3_comps = np.array(complex_t3_comps)

    # TODO: Make this implementation work again
    # for comp in components:
    #     if "Point" == comp.shortname:
    #         flux_ratio = comp.compute_flux(wavelength)
    #         if OPTIONS.model.output == "normed":
    #             stellar_flux = flux_ratio
    #         else:
    #             stellar_flux = (flux_model/(1-flux_ratio))*flux_ratio
    #
    #         flux_model += stellar_flux
    #         # complex_vis_model += stellar_flux
    #         # complex_t3_model += stellar_flux
    #         break

    # TODO: This summation is done twice, can it be reduced to once?
    flux_model = complex_vis_comps[..., 0].sum(axis=0).reshape(-1, 1)
    t3_model = compute_t3(complex_t3_comps.sum(axis=0))

    if OPTIONS.model.output == "normed":
        complex_vis_comps /= flux_model

    if flux_model.size > 0:
        flux_model = np.tile(flux_model, (len(OPTIONS.data.readouts))).real

    if not rzero:
        complex_vis_comps = complex_vis_comps[:, :, 1:]
        if t3_model.size > 0:
            t3_model = t3_model[:, 1:]

    vis_model = complex_vis_comps.sum(axis=0)
    if not rraw:
        vis_model = compute_vis(vis_model)

    if "vis2" in OPTIONS.fit.data:
        vis_model *= vis_model

    if rcomponents:
        vis_comps = compute_vis(complex_vis_comps)
        if "vis2" in OPTIONS.fit.data:
            vis_comps *= vis_comps

        return flux_model, vis_model, t3_model, vis_comps

    return flux_model, vis_model, t3_model


def compute_sed_chi_sq(flux_model: np.ndarray, ndim: int, method: str) -> float:
    """Calculates the sed model's chi square.

    Parameters
    ----------
    flux_model : numpy.ndarray
        The model's total flux.
    ndim : int, optional
        The number of (parameter) dimensions.
    method : str
        The method to determine the difference of the dataset,
        to the data. Either "linear" or "logarithmic".

    Returns
    -------
    chi_sq : float
        The chi square.
    """
    flux = OPTIONS.data.flux
    nan_indices = np.isnan(flux.value)
    chi_sq = compute_chi_sq(
        flux.value[~nan_indices],
        flux.err[~nan_indices],
        flux_model.flatten(),
        ndim,
        method=method,
    )

    # NOTE: The -1 here indicates that one of the parameters is actually fixed
    return chi_sq / (flux.value.size - ndim - 1)


def compute_interferometric_chi_sq(
    flux_model: np.ndarray,
    vis_model: np.ndarray,
    t3_model: np.ndarray,
    ndim: int,
    method: str,
    reduced: bool = False,
) -> Tuple[float, float, float, float]:
    """Calculates the disc model's chi square.

    Parameters
    ----------
    flux_model : numpy.ndarray
        The model's total flux.
    vis_model : numpy.ndarray
        Either the model's correlated fluxes or the model's
        visibilities (depends on the OPTIONS.fit.data).
    t3_model : numpy.ndarray
        The model's closure phase.
    ndim : int
        The number of (parameter) dimensions.
    method : bool
        The method used to calculate the chi square.
        Either "linear" or "logarithmic".
    reduced : bool, optional
        Whether to return the reduced chi square.

    Returns
    -------
    chi_sq : Tuple of floats
        The total and the individual chi squares.
    """
    params = {"flux": flux_model, "vis": vis_model, "t3": t3_model}
    weights = [getattr(OPTIONS.fit.weights, key) for key in OPTIONS.fit.data]

    chi_sqs = []
    for key in OPTIONS.fit.data:
        data = getattr(OPTIONS.data, key)
        nan_indices = np.isnan(data.value)
        chi_sq = compute_chi_sq(
            data.value[~nan_indices],
            data.err[~nan_indices],
            params[key if key != "vis2" else "vis"][~nan_indices],
            ndim=ndim,
            diff_method="linear" if key != "t3" else "exponential",
            method=method,
        )
        chi_sqs.append(chi_sq)

    ndata = get_counts_data()

    # TODO: Make sure the weights are correct for not only the vis, but
    # also the closure phase fit
    chi_sqs = np.array(chi_sqs).astype(float) * weights
    if reduced:
        total_chi_sq = chi_sqs.sum() / np.abs(ndata.sum() - ndim)
        chi_sqs = chi_sqs / np.abs(ndata - ndim)
    else:
        total_chi_sq = chi_sqs.sum()

    return (total_chi_sq, *chi_sqs)


def transform_uniform_prior(theta: List[float]) -> float:
    """Prior transform for uniform priors."""
    priors = get_priors()
    return priors[:, 0] + (priors[:, 1] - priors[:, 0]) * theta


# TODO: Same as below
def ptform_one_disc(theta: List[float], labels: List[str]) -> np.ndarray:
    """Transform that hard constrains the model to one continous disc by
    setting the outer radius of the first component to the inner of the second.

    Also makes a smooth continous transition of the surface density profiles.

    NOTES
    -----
    Only works with two components (as of now - could be extended).
    """
    params, priors = transform_uniform_prior(theta), get_priors()
    indices_radii = list(
        map(labels.index, (filter(lambda x: "rin" in x or "rout" in x, labels)))
    )
    params[indices_radii[2]] = params[indices_radii[1]]
    params[indices_radii[-1]] = (
        params[indices_radii[2]]
        + np.diff(priors[indices_radii[-1]])[0] * theta[indices_radii[-1]]
    )
    if params[indices_radii[-2]] > priors[indices_radii[-2]][1]:
        params[indices_radii[-2]] = priors[indices_radii[-2]][1]

    indices_sigma0 = list(map(labels.index, filter(lambda x: "sigma0" in x, labels)))
    indices_p = list(map(labels.index, filter(lambda x: "p" in x, labels)))
    r0 = OPTIONS.model.reference_radius.value
    sigma01, p1, p2 = (
        params[indices_sigma0[0]],
        params[indices_p[0]],
        params[indices_p[1]],
    )
    params[indices_sigma0[1]] = sigma01 * (params[indices_radii[1]] / r0) ** (p1 - p2)

    return params


# TODO: Fix this piece of code so it goes backwards to forwards. Start with the
# largest radii first, only if all fit in line, then do it, or?
# Smallest still ok if the largest one is capped?
# Do it more naturally? As in some transform of the radii? Not just clipping it?
# Use np.clip also
def ptform_sequential_radii(theta: List[float], labels: List[str]) -> np.ndarray:
    """Transform that soft constrains successive radii to be smaller than the one before."""
    params, priors = transform_uniform_prior(theta).tolist(), get_priors()
    radii = list(filter(lambda x: "rin" in x or "rout" in x, labels))
    if "rout" not in radii[-1]:
        radii.append(f"rout-{radii[-1].split('-')[1]}")

    indices = list(
        map(labels.index, (filter(lambda x: "rin" in x or "rout" in x, labels)))
    )
    params.append(OPTIONS.model.components_and_params[-1][1]["rout"].value)
    breakpoint()

    for count, index in enumerate(indices[::-1]):
        if count == len(indices) - 1:
            break

        current_radius, next_radius = params[index], params[indices[count + 1]]
        if next_radius <= current_radius:
            next_theta, next_priors = (
                theta[indices[count + 1]],
                priors[indices[count + 1]],
            )
            updated_radius = current_radius + np.diff(next_priors)[0] * next_theta

            if updated_radius > next_priors[1]:
                updated_radius = next_priors[1]

            params[indices[count + 1]] = updated_radius

    return params


def ptform_lband_fit(theta: List[float], labels: List[str]) -> np.ndarray:
    """Transform that soft constrains successive radii to be smaller than the one before."""
    indices = list(map(labels.index, filter(lambda x: "weight" in x, labels)))
    params = transform_uniform_prior(theta)

    remainder = 100
    for index in indices[:-1]:
        params[index] *= remainder / 1e2
        remainder = max(0, remainder - params[index])

    params[indices[-1]] = remainder if remainder > 1e-5 else 0
    return params


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
    parameters, shared_params = set_components_from_theta(theta)

    components = assemble_components(parameters, shared_params)
    return sum(
        compute_interferometric_chi_sq(
            *compute_observables(components), ndim=theta.size, method="logarithmic"
        )[1:]
    )


def lnprob_nband_fit(theta: np.ndarray) -> float:
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
    components = assemble_components(theta)
    return compute_sed_chi_sq(
        components[0].compute_flux(OPTIONS.fit.wavelengths),
        ndim=len(get_priors()),
        method="logarithmic",
    )


def run_fit(
    sample: str = "rwalk",
    bound: str = "multi",
    ncores: int = 6,
    debug: bool = False,
    method: str = "dynamic",
    save_dir: Path | None = None,
    **kwargs,
) -> NestedSampler | DynamicNestedSampler:
    """Runs the dynesty nested sampler.

    Parameters
    ----------
    sample : str, optional
        The sampling method. Either "rwalk" or "unif".
    bound : str, optional
        The bounding method. Either "multi" or "single".
    ncores : int, optional
        The number of cores to use.
    debug : bool, optional
        Whether to run the sampler in debug mode.
        This will not use multiprocessing.
    method : str, optional
        The method to use. Either "dynamic" or "static".
    save_dir : Path, optional
        The directory to save the sampler.

    Returns
    -------
    sampler : dynesty.NestedSampler or dynesty.DynamicNestedSampler
    """
    if save_dir is not None:
        checkpoint_file = save_dir / "sampler.save"
    else:
        checkpoint_file = None

    samplers = {"dynamic": DynamicNestedSampler, "static": NestedSampler}
    pool = Pool(processes=ncores) if not debug else None
    queue_size = ncores if not debug else None

    general_kwargs = {
        "bound": bound,
        "queue_size": queue_size,
        "sample": sample,
        "periodic": kwargs.pop("periodic", None),
        "reflective": kwargs.pop("reflective", None),
        "pool": pool,
    }

    # TODO: Check if update interval needs to be reimplemented
    # TODO: Check what periodic and reflective does
    static_kwargs = {"nlive": kwargs.pop("nlive", 1000)}
    sampler_kwargs = {"dynamic": {}, "static": static_kwargs}

    static = {"dlogz": kwargs.pop("dlogz", 0.01)}
    dynamic = {
        "nlive_batch": kwargs.pop("nlive_batch", 1000),
        "maxbatch": kwargs.pop("maxbatch", 100),
        "dlogz_init": kwargs.pop("dlogz_init", 0.01),
        "nlive_init": kwargs.pop("nlive_init", 1000),
    }
    run_kwargs = {"dynamic": dynamic, "static": static}

    print(f"Executing Dynesty.\n{'':-^50}")
    ptform = kwargs.pop("ptform", transform_uniform_prior)
    sampler = samplers[method](
        kwargs.pop("lnprob", lnprob),
        ptform,
        len(get_priors()),
        **general_kwargs,
        **sampler_kwargs[method],
    )
    sampler.run_nested(
        **run_kwargs[method], print_progress=True, checkpoint_file=str(checkpoint_file)
    )

    if not debug:
        pool.close()
        pool.join()
    return sampler


def get_best_fit(
    sampler: NestedSampler | DynamicNestedSampler,
    method: str = "max",
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Gets the best fit from the emcee sampler."""
    results = sampler.results
    weights = results.importance_weights()

    quantiles = np.array(
        [
            dyutils.quantile(
                samps, np.array(OPTIONS.fit.quantiles) / 100, weights=weights
            )
            for samps in results.samples.T
        ]
    )

    params = None
    if method == "max":
        params = results.samples[results.logl.argmax()]
    elif method == "quantile":
        params = quantiles[:, 1]

    uncertainties = np.array([(quantile[0], quantile[1]) for quantile in quantiles])

    return params, uncertainties
