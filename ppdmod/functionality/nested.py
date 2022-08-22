import dynesty
import numpy as np
import matplotlib.pyplot as plt

from schwimmbad import MPIPool
from multiprocessing import Pool, cpu_count
from dynesty import utils as dyfunc
from dynesty import plotting as dyplot
from typing import Any, Dict, List, Union, Optional, Callable

from .fitting_utils import lnlike, plot_fit_results


def plot_runs(results, save_path: Optional[str] = ""):
    """Similar to the chain-plot of mcmc, this displays the run of a dynesty fit"""
    rfig, raxes = dyplot.runplot(results)
    if save_path == "":
        plt.savefig(plot_name)
    else:
        plt.savefig(os.path.join(save_path, plot_name))

def plot_trace(results, save_path: Optional[str] = "") -> None:
    """Makes a traceplot for the result of the dynesty fit"""
    tfig, taxes = dyplot.traceplot(results, truths=np.zeros(ndim), truth_color="black",
                                 show_titles=True, trace_cmap="viridis", connect=True)
    if save_path == "":
        plt.savefig(plot_name)
    else:
        plt.savefig(os.path.join(save_path, plot_name))

def plot_corner(results, save_path: Optional[str] = ""):
    """This plots the corner plot for a dynesty fit"""
    cfig, caxes = dyplot.cornerplot(results, color="blue", truths=np.zeros(ndim),
                                    truth_color="black", show_titles=True,
                                    quantiles=None)
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
                labels: List, lnlike: Callable, data: List, plot_wl: List,
                frac: Optional[float] = 1e-4, cluster: Optional[bool] = False,
                method: Optional[str] = "dynamic",
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
    frac: float, optional
    cluster: bool, optional
    method: str, optional
    save_path: str, optional
    """
    initial, nlive, nlive_init = hyperparams
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
                                                       pool=pool, queue_size=cpu_amount)
            else:
                sampler = dynesty.NestedSampler(lnlike, ptform, ndim,
                                                logl_args=data,
                                                ptform_args=[priors],
                                                update_interval=float(ndim),
                                                sample="rwalk", bound="multi",
                                                nlive=nlive,
                                                pool=pool, queue_size=cpu_amount)

            sampler.run_nested(dlogz=0.001, print_progress=True)

    results = sampler.results
    samples, weights = results.samples, np.exp(results.logwt - results.logz[-1])
    mean, cov = dyfunc.mean_and_cov(samples, weights)
    new_samples = dyfunc.resample_equal(samples, weights)

    plot_trace(results, save_path)
    plot_runs(results, save_path)
    plot_corner(results, save_path)
    # plot_fit_results(save_path)
    plt.show()


if __name__ == "__main__":
    ...

