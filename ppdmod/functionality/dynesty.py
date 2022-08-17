import dynesty
import numpy as np

from schwimmbad import MPIPool
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, List, Union, Optional, Callable


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

    transformed_priors = []
    for i, o in enumerate(priors):
        transformed_priors.append(uniform_transform(initial[i], *o))

    return transformed_priors

def run_dynesty(hyperparams: List, priors: List,
                labels: List, lnprob: Callable, data: List, plot_wl: List,
                frac: Optional[float] = 1e-4, cluster: Optional[bool] = False,
                debug: Optional[bool] = False,
                save_path: Optional[str] = "") -> np.array:
    """Runs the dynesty nested sampler

    The (Dynamic)NestedSampler recieves the parameters and the data are
    reformatted and passed to the 'lnprob' method

    Parameters
    ----------
    hyperparams: List
    priors: List
    labels: List
    lnprob: Callable
    data: List
    plot_wl: float
    frac: float, optional
    cluster: bool, optional
    debug: bool, optional
    save_path: str, optional
    """
    initial, nlive = hyperparams
    ndim = len(inital)

    if cluster:
        with MPIPool as pool:
            if not pool.is_master():
                pool.wait()
                sys.exit(0)

    else:
        with Pool() as pool:
            print(f"Executing DYNESTY with {cpu_count()} cores.")

        ndim = len(p0)

        if method == "dynamic":
            sampler = dynesty.DynamicNestedSampler(lnlike, ptform, ndim,
                                                   logl_args=ln_args,
                                                   ptform_args=ptf_args,
                                                   update_interval=float(ndim),
                                                   pool=pool)
        else:
            sampler = dynesty.NestedSampler(lnlike, ptform, )
            sampler = dynesty.NestedSampler(lnlike, ptform, ndim,
                                            logl_args=ln_args,
                                            ptform_args=ptf_args,
                                            update_interval=float(ndim),
                                            pool=pool)

        sampler.run_nested(progress=True)

        samples, weights = results.samples,\
                np.exp(results.logwt - results.logz[-1])
        mean, cov = dyfunc.mean_and_cov(samples, weights)
        new_samples = dyfunc.resample_equal(samples, weights)


if __name__ == "__main__":
    print(ptform([0, 3], [[1, 2], [0, 5]]))
