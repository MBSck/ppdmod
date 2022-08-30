from pathlib import Path
from typing import Any, Dict, List, Union, Optional, Callable

from ppdmod.models import *
from ppdmod.functionality.mcmc import run_mcmc
from ppdmod.functionality.nested import run_dynesty, ptform
from ppdmod.functionality.synthetic import create_model
from ppdmod.functionality.baseClasses import Model
from ppdmod.functionality.fitting_utils import get_rndarr_from_bounds,\
        get_data_for_fit, lnprob, lnlike

# TODO: Add docstrings to the big overhead functions in this script
# TODO: Improve code with namedtuples ASAP

def create_synthetic_dataset(priors: List, bb_params: List, model: Model, fov_size: int,
                             px_sampling: int, zero_padding_order: Optional[int] = 2,
                             intp: Optional[bool] = True, fits_file: Optional[Path] = "",
                             path_to_fits: Optional[Path] = "",
                             save_path: Optional[str] = "") -> None:
    """Creates a synthetic dataset by looping over the model for a certain parameter
    input

    Parameters
    ----------
    priors: List
    bb_params: List
    path_to_fits: Path, optional
    fits_file: Path, optional
    """
    theta = get_rndarr_from_bounds(priors)
    create_model(save_path, model, theta, bb_params, fov_size, px_sampling,
                 path_to_fits=path_to_fits, fits_file=fits_file)

def run_mcmc_fit(priors: List, labels: List,
                 bb_params: List, mcmc_params: List,
                 wl_sel: List, path_to_fits: Path,
                 model: Model, pixel_size: int, sampling: int,
                 zero_padding_order: int, frac: Optional[float] = 1e-4,
                 cluster: Optional[bool] = False, synthetic: Optional[bool] = False,
                 save_path: Optional[Path] = "", vis2: Optional[bool] = False,
                 intp: Optional[bool] = True, flux_file: Optional[Path] = None,
                 initial: Optional[List] = None) -> None:
    """Executes a mcmc-fit"""
    data = get_data_for_fit(CompoundModel, pixel_size=pixel_size,
                            sampling=sampling, wl_sel=wl_sel,
                            flux_file=flux_file,
                            zero_padding_order=zero_padding_order,
                            bb_params=bb_params, priors=priors, vis2=vis2,
                            intp=intp, path_to_fits=path_to_fits)

    run_mcmc(mcmc_params, priors, labels, lnprob, data, plot_wl=wl_sel,
             frac=frac, cluster=cluster, synthetic=synthetic, save_path=save_path)

def run_nested_fit(priors: List, labels: List,
                   bb_params: List, dynesty_params: List,
                   wl_sel: List, path_to_fits: Path,
                   model: Model, pixel_size: int, sampling: int,
                   zero_padding_order: int, method: Optional[str] = "dynamic",
                   sampling_method: Optional[str] = "auto",
                   bounding_method: Optional[str] = "multi",
                   frac: Optional[float] = 1e-4,
                   cluster: Optional[bool] = False, synthetic: Optional[bool] = False,
                   save_path: Optional[Path] = "", vis2: Optional[bool] = False,
                   intp: Optional[bool] = True, flux_file: Optional[Path] = None,
                   initial: Optional[List] = None) -> None:
    """Executes a dynesty-fit"""
    data = get_data_for_fit(CompoundModel, pixel_size=pixel_size,
                            sampling=sampling, wl_sel=wl_sel,
                            flux_file=flux_file,
                            zero_padding_order=zero_padding_order,
                            bb_params=bb_params, priors=priors, vis2=vis2,
                            intp=intp, path_to_fits=path_to_fits)

    run_dynesty(dynesty_params, priors, labels, lnlike, data, wl_sel,
                synthetic=synthetic, frac=frac, cluster=cluster,
                method=method, sampling_method=sampling_method,
                bounding_method=bounding_method, save_path=save_path)


if __name__ == "__main__":
    priors = [[1., 2.], [0, 180], [0.5, 1.], [0, 360], [1., 10.], [0., 1.], [0., 1.]]
    initial = get_rndarr_from_bounds(priors, True)
    labels = ("axis ratio", "pos angle", "mod amplitude",
              "mod angle", "inner radius", "tau", "q")
    wl_sel = (8.5, 10., 12.5)
    bb_params = (1500, 9200, 16, 101.2)

    mcmc_params = (initial, 32, 250, 500)
    dynesty_params = (initial, 15)

    path_to_fits = "assets/data/HD_142666/test"
    save_path = "assets/data/SyntheticModels/"
    # create_synthetic_dataset(priors, bb_params, CompoundModel, 50, 1024,
                             # path_to_fits=path_to_fits, save_path=save_path)

    path_to_fits = save_path
    save_path = "assets/model_results"
    flux_file = None

    # run_nested_fit(priors, labels, bb_params, dynesty_params, wl_sel,
                   # path_to_fits, CompoundModel, 50, 128, 1,
                   # method="static", sampling_method="unif",
                   # bounding_method="multi", synthetic=True)

    run_mcmc_fit(priors, labels, bb_params, mcmc_params, wl_sel, path_to_fits,
                 CompoundModel, 30, 128, 2)

