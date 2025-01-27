from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple

import astropy.units as u
import dynesty.utils as dyutils
import emcee
import numpy as np
from dynesty import DynamicNestedSampler

from .component import Component
from .data import get_counts_data
from .options import OPTIONS
from .utils import compare_angles, compute_t3, compute_vis


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
        to the data. Either "linear" or "periodic".
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
    sn = error**2
    if lnf is not None:
        sn += model_data**2 * np.exp(2 * lnf)

    residuals = data - model_data
    if diff_method == "periodic":
        residuals = np.rad2deg(compare_angles(np.deg2rad(data), np.deg2rad(model_data)))

    chi_sq = residuals**2 / sn
    if method == "linear":
        return chi_sq.sum()

    if OPTIONS.fit.fitter == "dynesty":
        chi_sq += np.log(sn) + data.size * np.log(2 * np.pi)
    return -0.5 * np.sum(chi_sq)


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
    components: List[Component],
    ndim: int,
    method: str,
    reduced: bool = False,
) -> Tuple[float, float, float, float]:
    """Calculates the disc model's chi square.

    Parameters
    ----------
    components : list of Component
        The components to be used in the model.
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
    observables = ["flux", "vis", "t3"]
    model_data = dict(zip(observables, compute_observables(components)))
    weights = [getattr(OPTIONS.fit.weights, key) for key in OPTIONS.fit.data]

    chi_sqs = []
    for key in OPTIONS.fit.data:
        data = getattr(OPTIONS.data, key)
        key = key if key != "vis2" else "vis"
        mask = data.value.mask
        chi_sqs.append(
            compute_chi_sq(
                data.value.data[~mask],
                data.err.data[~mask],
                model_data[key][~mask],
                diff_method="linear" if key != "t3" else "periodic",
                method=method,
                lnf=getattr(components[-1], f"{key}_lnf")(),
            )
        )

    chi_sqs = np.array(chi_sqs).astype(float)

    if reduced:
        ndata = get_counts_data()
        total_chi_sq = chi_sqs.sum() / np.abs(ndata.sum() - ndim)
        chi_sqs /= np.abs(ndata - ndim)
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

    indices = list(
        map(labels.index, (filter(lambda x: "rin" in x or "rout" in x, labels)))
    )

    new_radii = [params[indices][0]]
    for index, (uniform, prior) in enumerate(
        zip(theta[indices][1:], priors[indices][1:]), start=1
    ):
        prior[0] = new_radii[index - 1]
        new_radii.append(prior[0] + (prior[1] - prior[0]) * uniform)

    params[indices] = new_radii
    return params


def lnprior(components: List[Component]) -> float:
    """Checks if the parameters are within the priors (for emcee)."""
    for param, prior in zip(get_theta((components)), get_priors(components)):
        if not prior[0] <= param <= prior[1]:
            return -np.inf
    return 0.0


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
    if OPTIONS.fit.fitter == "emcee":
        if np.isinf(lnprior(components)):
            return -np.inf

    return compute_interferometric_chi_sq(
        components, ndim=theta.size, method="logarithmic", reduced=True
    )[0]



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


# TODO: Init this correctly
def init_uniformly(nwalkers: int, labels: List[str]) -> np.ndarray:
    """initialises a random numpy.ndarray from the parameter's limits.

    Parameters
    -----------
    nwalkers : list of list of dict

    Returns
    -------
    theta : numpy.ndarray
    """
    priors = get_priors(OPTIONS.model.components)
    # TODO: This could be made more general
    indices = list(
        map(labels.index, (filter(lambda x: "rin" in x or "rout" in x, labels)))
    )
    uniform_grid = np.array(
        [
            [np.random.uniform(0, 1) for _ in range(len(priors))]
            for _ in range(nwalkers)
        ]
    )
    sample_grid = np.array([
        prior[0] + (prior[1] - prior[0]) * uniform_grid[:, i]
        for i, prior in enumerate(priors)
    ]).T
    for row_index, (value_row, uniform_row) in enumerate(zip(sample_grid, uniform_grid)):
        new_radii = [value_row[indices][0]]
        for index, (uniform, prior) in enumerate(
            zip(uniform_row[indices][1:], priors[indices][1:]), start=1
        ):
            prior[0] = new_radii[index - 1]
            new_radii.append(prior[0] + (prior[1] - prior[0]) * uniform)

        sample_grid[row_index, indices] = new_radii
    return sample_grid


def run_emcee(
    init_guess: np.ndarray,
    nwalkers: int,
    nburnin: int = 0,
    nsteps: int = 100,
    ncores: int = 6,
    debug: bool = False,
    save_dir: Path | None = None,
    **kwargs,
) -> emcee.EnsembleSampler:
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
    pool = Pool(processes=ncores) if not debug else None

    print(f"Executing MCMC.\n{'':-^50}")
    sampler = emcee.EnsembleSampler(
        nwalkers, init_guess.shape[1], kwargs.pop("lnprob", lnprob), pool=pool
    )

    if nburnin is not None:
        print("Running burn-in...")
        sampler.run_mcmc(init_guess, nburnin, progress=True)

    sampler.reset()
    print("Running production...")
    sampler.run_mcmc(init_guess, nsteps, progress=True)

    if save_dir is not None:
        np.save(save_dir / "sampler.npy", sampler)

    if not debug:
        pool.close()
        pool.join()
    return sampler


def run_dynesty(
    sample: str = "rwalk",
    bound: str = "multi",
    ncores: int = 6,
    debug: bool = False,
    save_dir: Path | None = None,
    **kwargs,
) -> DynamicNestedSampler:
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
    save_dir : Path, optional
        The directory to save the sampler.

    Returns
    -------
    sampler : dynesty.DynamicNestedSampler
    """
    if save_dir is not None:
        checkpoint_file = save_dir / "sampler.save"
    else:
        checkpoint_file = None

    pool = Pool(processes=ncores) if not debug else None
    queue_size = 2 * ncores if not debug else None

    general_kwargs = {
        "bound": bound,
        "queue_size": queue_size,
        "sample": sample,
        "periodic": kwargs.pop("periodic", None),
        "reflective": kwargs.pop("reflective", None),
        "pool": pool,
    }

    run_kwargs = {
        "nlive_batch": kwargs.pop("nlive_batch", 500),
        "dlogz_init": kwargs.pop("dlogz_init", 0.01),
        "nlive_init": kwargs.pop("nlive_init", 1000),
    }

    print(f"Executing Dynesty.\n{'':-^50}")
    ndim = len(get_priors(OPTIONS.model.components))
    ptform = kwargs.pop("ptform", transform_uniform_prior)
    sampler = DynamicNestedSampler(
        kwargs.pop("lnprob", lnprob),
        ptform,
        ndim,
        **general_kwargs,
    )
    sampler.run_nested(
        **run_kwargs, print_progress=True, checkpoint_file=str(checkpoint_file)
    )

    if not debug:
        pool.close()
        pool.join()
    return sampler


def run_fit(**kwargs):
    """Runs the fit."""
    if OPTIONS.fit.fitter == "emcee":
        return run_emcee(**kwargs)
    return run_dynesty(**kwargs)


def get_best_fit(
    sampler: DynamicNestedSampler,
    method: str = "max",
) -> Tuple[np.ndarray, np.ndarray]:
    """Gets the best fit from the emcee sampler."""
    if OPTIONS.fit.fitter == "emcee":
        samples = sampler.get_chain(flat=True)
        quantiles = np.percentile(samples, OPTIONS.fit.quantiles, axis=0)
        if method == "max":
            quantiles[1] = samples[np.argmax(sampler.get_log_prob(flat=True))]
    else:
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

        if method == "max":
            quantiles[:, 1] = results.samples[results.logl.argmax()]

    return quantiles[1], np.diff(quantiles, axis=0)
