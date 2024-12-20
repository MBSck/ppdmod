from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple

import astropy.units as u
import dynesty.utils as dyutils
import numpy as np
from dynesty import DynamicNestedSampler, NestedSampler

from .component import Component
from .data import get_counts_data
from .options import OPTIONS
from .utils import compute_vis, compute_t3, compare_angles, get_band


def get_labels(components: List[Component], shared: bool = True) -> np.ndarray:
    """Sets theta from the components.

    Parameters
    ----------
    components : list of Component
        The components to be used in the model.
    shared : bool, optional
        If true, gets the shared params from the components

    Returns
    -------
    theta : numpy.ndarray
    """
    disc_models = ["TempGrad", "AsymTempGrad", "GreyBody", "AsymGreyBody"]
    labels, labels_shared, zone_count = [], [], 1
    for component in components:
        component_labels = [key for key in component.get_params(free=True)]
        if component.name == "Star":
            component_labels = [rf"{label}-\star" for label in component_labels]

        if component.name in disc_models:
            component_labels = [f"{label}-{zone_count}" for label in component_labels]
            zone_count += 1

        labels.extend(component_labels)

        if shared:
            labels_shared.append(
                [rf"{key}-\mathrm{{sh}}" for key in component.get_params(shared=True)]
            )

    labels.extend(labels_shared[-1])
    return labels


def get_priors(components: List[Component], shared: bool = True) -> np.ndarray:
    """Gets the priors from the model parameters."""
    priors, priors_shared = [], []
    for component in components:
        priors.extend(
            [param.get_limits() for param in component.get_params(free=True).values()]
        )

        if shared:
            priors_shared.append(
                [
                    param.get_limits()
                    for param in component.get_params(shared=True).values()
                ]
            )
    priors.extend(priors_shared[-1])
    return np.array(priors)


def get_units(
    components: List[Component],
    shared: bool = True,
) -> np.ndarray:
    """Sets the units from the components.

    Parameters
    ----------
    components : list of Component
        The components to be used in the model.
    shared_params : dict
        The shared parameters.
    shared : bool, optional
        If true, gets the shared params from the components

    Returns
    -------
    units : numpy.ndarray
    """
    units, units_shared = [], []
    for component in components:
        units.extend([param.unit for param in component.get_params(free=True).values()])

        if shared:
            units_shared.append(
                [param.unit for param in component.get_params(shared=True).values()]
            )

    units.extend(units_shared[-1])
    return np.array(units)


def get_theta(
    components: List[Component],
    shared: bool = True,
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
    theta, theta_shared = [], []
    for component in components:
        theta.extend(
            [param.value for param in component.get_params(free=True).values()]
        )

        if shared:
            theta_shared.append(
                [param.value for param in component.get_params(shared=True).values()]
            )

    theta.extend(theta_shared[-1])
    return np.array(theta)


def set_components_from_theta(theta: np.ndarray) -> List[Component]:
    """Sets the components from theta."""
    components = [component.copy() for component in OPTIONS.model.components]
    nshared = len(components[-1].get_params(shared=True))
    if nshared != 0:
        theta_list, shared_params = theta[:-nshared], theta[-nshared:]
    else:
        theta_list, shared_params = theta, []

    theta_list = theta_list.copy().tolist()
    shared_params_labels = [
        label.split("-")[0] for label in get_labels(components) if "sh" in label
    ]

    for component in components:
        for param in component.get_params(free=True).values():
            param.value = theta_list.pop(0)
            param.free = True

        for param_name, value in zip(shared_params_labels, shared_params):
            if hasattr(component, param_name):
                param = getattr(component, param_name)
                param.value = value
                param.shared = True

    return components


def compute_chi_sq(
    data: u.Quantity,
    error: u.Quantity,
    model_data: u.Quantity,
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
        diff = np.rad2deg(compare_angles(np.deg2rad(data), np.deg2rad(model_data)))

    chi_sq = diff**2 / sigma_squared
    if method == "linear":
        return chi_sq.sum()

    return -0.5 * (chi_sq + np.log(2 * np.pi * sigma_squared)).sum()


def compute_observables(
    components: List[Component],
    wavelength: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculates the observables from the model.

    Parameters
    ----------
    components : list of Component
        The components to be used in the model.
    wavelength : numpy.ndarray, optional
        The wavelength to be used in the model.
    """
    wavelength = OPTIONS.fit.wavelengths if wavelength is None else wavelength
    vis = OPTIONS.data.vis2 if "vis2" in OPTIONS.fit.data else OPTIONS.data.vis
    t3 = OPTIONS.data.t3
    complex_vis = np.sum(
        [
            comp.compute_complex_vis(vis.ucoord, vis.vcoord, wavelength)
            for comp in components
        ],
        axis=0,
    )
    complex_t3 = np.sum(
        [
            comp.compute_complex_vis(t3.ucoord, t3.vcoord, wavelength)
            for comp in components
        ],
        axis=0,
    )

    t3_model = compute_t3(complex_t3[:, t3.index123])
    flux_model = complex_vis[:, 0].reshape(-1, 1)

    if OPTIONS.model.output == "normed":
        complex_vis /= flux_model

    if "vis2" in OPTIONS.fit.data:
        complex_vis *= complex_vis

    complex_vis = compute_vis(complex_vis[:, 1:])
    if flux_model.size > 0:
        flux_model = np.tile(flux_model, OPTIONS.data.flux.value.shape[-1]).real

    return flux_model, complex_vis, t3_model


def compute_nband_fit_chi_sq(
    flux_model: np.ndarray,
    ndim: int,
    method: str,
    reduced: bool = False,
) -> float:
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
    reduced : bool, optional
        Whether to return the reduced chi square.

    Returns
    -------
    chi_sq : float
        The chi square.
    """
    # NOTE: The -1 here indicates that one of the parameters is actually fixed
    ndim -= 1
    flux = OPTIONS.data.flux
    mask = flux.value.mask
    chi_sq = compute_chi_sq(
        flux.value.data[~mask].astype(OPTIONS.data.dtype.real),
        flux.err.data[~mask].astype(OPTIONS.data.dtype.real),
        flux_model[~mask],
        method=method,
    )

    if reduced:
        return chi_sq / (flux.value.size - ndim)

    return chi_sq


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

    chi_sqs, weights = [], []
    for key in OPTIONS.fit.data:
        data = getattr(OPTIONS.data, key)
        key = key if key != "vis2" else "vis"
        weights.append(getattr(OPTIONS.fit.weights, key))
        mask = data.value.mask
        chi_sqs.append(
            compute_chi_sq(
                data.value.data[~mask],
                data.err.data[~mask],
                params[key][~mask],
                diff_method="linear" if key != "t3" else "exponential",
                method=method,
            )
        )

    chi_sqs = np.array(chi_sqs).astype(float)

    if reduced:
        ndata = get_counts_data()
        total_chi_sq = chi_sqs.sum() / np.abs(ndata.sum() - ndim)
        chi_sqs = chi_sqs / np.abs(ndata - ndim)
    else:
        chi_sqs *= weights
        total_chi_sq = chi_sqs.sum()

    return (total_chi_sq, *chi_sqs)


def transform_uniform_prior(theta: List[float]) -> float:
    """Prior transform for uniform priors."""
    priors = get_priors(OPTIONS.model.components)
    return priors[:, 0] + (priors[:, 1] - priors[:, 0]) * theta


def ptform_nband_fit(theta: List[float], labels: List[str]) -> np.ndarray:
    """Transform that soft constrains successive radii to be smaller than the one before."""
    indices = list(map(labels.index, filter(lambda x: "weight" in x, labels)))
    params = transform_uniform_prior(theta)

    remainder = 100
    for index in indices[:-1]:
        params[index] = remainder * theta[index]
        remainder -= params[index]

    params[indices[-1]] = remainder
    return params


# TODO: Improve this and make it work again
def ptform_one_disc(theta: List[float], labels: List[str]) -> np.ndarray:
    """Transform that hard constrains the model to one continous disc by
    setting the outer radius of the first component to the inner of the second.

    Also makes a smooth continous transition of the surface density profiles.

    NOTES
    -----
    Only works with two components (as of now - could be extended).
    """
    params = (transform_uniform_prior(theta),)
    priors = get_priors(OPTIONS.model.components)
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


def ptform_sequential_radii(theta: List[float], labels: List[str]) -> np.ndarray:
    """Transform that soft constrains successive radii to be smaller than the one before."""
    priors = get_priors(OPTIONS.model.components)
    params = transform_uniform_prior(theta)

    radii_labels = list(filter(lambda x: "rin" in x or "rout" in x, labels))
    indices = list(
        map(labels.index, (filter(lambda x: "rin" in x or "rout" in x, labels)))
    )

    radii_values = params[indices].copy().tolist()
    radii_uniforms = theta[indices].copy().tolist()
    radii_priors = priors[indices].copy().tolist()

    if "rout" not in radii_labels[-1]:
        radii_labels.append(f"rout-{radii_labels[-1].split('-')[1]}")
        radii_values.append(OPTIONS.model.components[-1].rout.value)
        radii_uniforms.append(1)
        radii_priors.append([0, 0])

    radii_values = radii_values[::-1]
    radii_uniforms, radii_priors = radii_uniforms[::-1], radii_priors[::-1]

    new_radii = [radii_values[0]]
    for index, (radius, uniform, prior) in enumerate(
        zip(radii_values[1:], radii_uniforms[1:], radii_priors[1:]), start=1
    ):
        prior[-1] = new_radii[index - 1]
        new_radii.append(prior[0] + (prior[1] - prior[0]) * uniform)

    new_radii = new_radii[::-1]
    for index, radius in zip(indices, new_radii):
        params[index] = radius

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
    components = set_components_from_theta(theta)
    observables = compute_observables(components)
    return sum(
        compute_interferometric_chi_sq(
            *observables, ndim=theta.size, method="logarithmic"
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
    components = set_components_from_theta(theta)
    return compute_nband_fit_chi_sq(
        components[0].compute_flux(OPTIONS.fit.wavelengths),
        ndim=theta.size,
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

    static_kwargs = {"nlive": kwargs.pop("nlive", 1000)}
    sampler_kwargs = {"dynamic": {}, "static": static_kwargs}

    static = {"dlogz": kwargs.pop("dlogz", 0.01)}
    dynamic = {
        "nlive_batch": kwargs.pop("nlive_batch", 500),
        "dlogz_init": kwargs.pop("dlogz_init", 0.01),
        "nlive_init": kwargs.pop("nlive_init", 1000),
    }
    run_kwargs = {"dynamic": dynamic, "static": static}

    print(f"Executing Dynesty.\n{'':-^50}")
    ndim = len(get_priors(OPTIONS.model.components))
    ptform = kwargs.pop("ptform", transform_uniform_prior)
    sampler = samplers[method](
        kwargs.pop("lnprob", lnprob),
        ptform,
        ndim,
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
