import dynesty
import numpy as np
import matplotlib.pyplot as plt

from warnings import warn
from schwimmbad import MPIPool
from multiprocessing import Pool, cpu_count
from dynesty import utils as dyfunc
from dynesty import plotting as dyplot
from typing import Any, Dict, List, Union, Optional, Callable

from .fitting_utils import lnlike, plot_fit_results


def get_median_and_errors(samples, weights,
                          ndim: int, quantiles: List):
    """Gets the medians and the error bounds for the samples

    Parameters
    ----------
    samples
    ndim: int

    Returns
    medians
    errors_lower
    errors_upper
    """
    medians, errors_lower, errors_upper = [], [], []
    get_quantiles = lambda x, y: np.percentile(x, [100. * q for q in y])

    for i in range(ndim):
        q = get_quantiles(samples[:, i], quantiles)
        medians.append(q[1])
        errors_lower.append(abs(q[1] - q[0]))
        errors_upper.append(abs(q[2] - q[1]))

    return medians, errors_lower, errors_upper

def plot_runs(results, save_path: Optional[str] = ""):
    """Similar to the chain-plot of mcmc, this displays the run of a dynesty fit"""
    rfig, raxes = dyplot.runplot(results)

    plot_name = "Dynesty_runs_plot.png"

    if save_path == "":
        plt.savefig(plot_name)
    else:
        plt.savefig(os.path.join(save_path, plot_name))

def plot_trace(results, ndim: int, save_path: Optional[str] = "") -> None:
    """Makes a traceplot for the result of the dynesty fit"""
    tfig, taxes = dyplot.traceplot(results, truths=np.zeros(ndim), truth_color="black",
                                 show_titles=True, trace_cmap="viridis", connect=True)

    plot_name = "Dynesty_trace_plot.png"

    if save_path == "":
        plt.savefig(plot_name)
    else:
        plt.savefig(os.path.join(save_path, plot_name))

def plot_corner(results, ndim: int,
                labels: List, quantiles: List,
                save_path: Optional[str] = ""):
    """This plots the corner plot for a dynesty fit"""
    cfig, caxes = dyplot.cornerplot(results, color="blue", truths=np.zeros(ndim),
                                    truth_color="black", show_titles=True,
                                    quantiles=quantiles, labels=labels)

    plot_name = "Dynesty_corner_plot.png"

    if save_path == "":
        plt.savefig(plot_name)
    else:
        plt.savefig(os.path.join(save_path, plot_name))

def ptform(initial: List, priors: List) -> List:
    """Tranforms all the priors to uniforms

    Parameters
    ----------
    initial: List
        The initial fit parameters
    priors: List
        The priors for the fit

    Returns
    -------
    List
        The reformatted priors
    """
    uniform_transform = lambda x, y, z: x + (y-x)*z
    transformed_priors = [uniform_transform(*o, initial[i]) for i, o in enumerate(priors)]
    return np.array(transformed_priors, dtype=float)

def run_dynesty(hyperparams: List, priors: List,
                labels: List, lnlike: Callable,
                data: List, plot_wl: List,
                quantiles: Optional[List] = [0.16, 0.5, 0.84],
                frac: Optional[float] = 1e-4,
                cluster: Optional[bool] = False,
                method: Optional[str] = "dynamic",
                sampling_method: Optional[str] = "auto",
                bounding_method: Optional[str] = "multi",
                synthetic: Optional[bool] = False,
                save_path: Optional[str] = "") -> np.array:
    """Runs the dynesty nested sampler

    The (Dynamic)NestedSampler recieves the parameters and the data are
    reformatted and passed to the 'lnprob' method

    Parameters
    ----------
    hyperparams: List
    priors: List
    labels: List
    lnlike: Callable
    data: List
    plot_wl: float
    quantiles: List, optional
    frac: float, optional
    cluster: bool, optional
    method: str, optional
    sampling_method: str, optional
    bounding_method: str, optional
    synthetic: bool, optional
    save_path: str, optional
    """
    if synthetic:
        try:
            print("Loaded perfect parameters from the synthetic dataset")
            print(np.load("assets/theta.npy"))
        except FileNotFoundError:
            warn("No 'theta.npy' file could be located!", category=FileNotFoundError)
        finally:
            print("File search done.")

    initial, nlive = hyperparams
    ndim = len(initial)

    if cluster:
        with MPIPool as pool:
            if not pool.is_master():
                pool.wait()
                sys.exit(0)

    else:
        with Pool() as pool:
            cpu_amount = cpu_count()
            print(f"Executing DYNESTY with {cpu_amount} cores.")
            print("--------------------------------------------------------------")

            if method == "dynamic":
                sampler = dynesty.DynamicNestedSampler(lnlike, ptform, ndim,
                                                       logl_args=data,
                                                       ptform_args=[priors],
                                                       update_interval=float(ndim),
                                                       sample=sampling_method,
                                                       bound=bounding_method,
                                                       nlive=nlive, pool=pool,
                                                       queue_size=cpu_amount)
            else:
                sampler = dynesty.NestedSampler(lnlike, ptform, ndim,
                                                logl_args=data,
                                                ptform_args=[priors],
                                                update_interval=float(ndim),
                                                sample=sampling_method,
                                                bound=bounding_method,
                                                nlive=nlive, pool=pool,
                                                queue_size=cpu_amount)

            sampler.run_nested(dlogz=1., print_progress=True)

    results = sampler.results
    samples, weights = results.samples, np.exp(results.logwt - results.logz[-1])
    mean, cov = dyfunc.mean_and_cov(samples, weights)
    new_samples = dyfunc.resample_equal(samples, weights)

    medians, lower_errors, upper_errors = get_median_and_errors(new_samples, weights,
                                                                ndim, quantiles)

    plot_runs(results, save_path)
    plot_trace(results, ndim, save_path)
    plot_corner(results, ndim, labels, quantiles, save_path)
    plot_fit_results(medians, *data, hyperparams, labels,
                    plot_wl=plot_wl, save_path=save_path)


if __name__ == "__main__":
    ...

