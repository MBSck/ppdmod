import dynesty
import numpy as np
import matplotlib.pyplot as plt

from warnings import warn
from dynesty import utils as dyfunc
from dynesty import plotting as dyplot
from multiprocessing import Pool, cpu_count
from typing import List, Optional, Callable

from ppdmod.functionality.data_prep import DataHandler

from .fitting_utils import lnlike, plot_fit_results


def get_median_and_errors(samples, weights, ndim: int, quantiles: List):
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

def ptform(uniform: List, priors: List) -> List:
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
    transformed_priors = [uniform_transform(*o, uniform[i]) for i, o in enumerate(priors)]
    return np.array(transformed_priors, dtype=float)

def run_dynesty(data: DataHandler,
                quantiles: Optional[List] = [0.16, 0.5, 0.84],
                cpu_amount: Optional[int] = 6,
                save_path: Optional[str] = "") -> None:
    """Runs the dynesty nested sampler
    The (Dynamic)NestedSampler recieves the parameters and the data are
    reformatted and passed to the 'lnprob' method

    Parameters
    ----------
    lnlike: Callable
    data: DataHandler
    quantiles: List, optional
    save_path: str, optional
    """
    with Pool(processes=cpu_amount) as pool:
        print(f"Executing DYNESTY with {cpu_amount} cores.")
        print("--------------------------------------------------------------")

        if data.dynesty.method == "dynamic":
            sampler = dynesty.DynamicNestedSampler(lnlike, ptform, data.dynesty.ndim,
                                                   logl_args=[data],
                                                   ptform_args=[data.priors],
                                                   update_interval=float(data.dynesty.ndim),
                                                   sample=data.dynesty.sampling_method,
                                                   bound=data.dynesty.bounding_method,
                                                   nlive=data.dynesty.nlive, pool=pool,
                                                   queue_size=cpu_amount)
        else:
            sampler = dynesty.NestedSampler(lnlike, ptform, data.dynesty.ndim,
                                            logl_args=data,
                                            ptform_args=[data.priors],
                                            update_interval=float(data.dynesty.ndim),
                                            sample=data.dynesty.sampling_method,
                                            bound=data.dynesty.bounding_method,
                                            nlive=data.dynesty.nlive, pool=pool,
                                            queue_size=cpu_amount)

        sampler.run_nested(dlogz=1., print_progress=True)

    results = sampler.results
    samples, weights = results.samples, np.exp(results.logwt - results.logz[-1])
    mean, cov = dyfunc.mean_and_cov(samples, weights)
    new_samples = dyfunc.resample_equal(samples, weights)

    medians, lower_errors, upper_errors = get_median_and_errors(new_samples, weights,
                                                                data.dynesty.ndim,
                                                                quantiles)

    plot_runs(results, save_path=save_path)
    plot_trace(results, data.dynesty.ndim, save_path=save_path)
    plot_corner(results, data.dynesty.ndim, data.labels, quantiles, save_path=save_path)
    plot_fit_results(medians, data, save_path=save_path)


if __name__ == "__main__":
    ...
