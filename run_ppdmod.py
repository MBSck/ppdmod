from pathlib import Path
from typing import Any, Dict, List, Union, Optional, Callable

from ppdmod.models import *
from ppdmod.functionality.mcmc import run_mcmc
from ppdmod.functionality.nested import run_dynesty
from ppdmod.functionality.baseClasses import Model
from ppdmod.functionality.fitting_utils import get_rndarr_from_bounds,\
        get_data_for_fit, lnprob, lnlike


def run_mcmc_fit(priors: List, labels: List,
                 bb_params: List, mcmc_params: List,
                 wl_sel: List, path_to_fits: Path,
                 model: Model, pixel_size: int, sampling: int,
                 zero_padding_order: int, frac: Optional[float] = 1e-4,
                 cluster: Optional[bool] = False,
                 save_path: Optional[Path] = "", vis2: Optional[bool] = False,
                 intp: Optional[bool] = True, flux_file: Optional[Path] = None,
                 initial: Optional[List] = None) -> None:
    """Executes an mcmc-fit"""
    data = get_data_for_fit(CompoundModel, pixel_size=pixel_size,
                            sampling=sampling, wl_sel=wl_sel,
                            flux_file=flux_file,
                            zero_padding_order=zero_padding_order,
                            bb_params=bb_params, priors=priors, vis2=vis2,
                            intp=intp, path_to_fits=path_to_fits)

    run_mcmc(mcmc_params, priors, labels, lnprob, data, plot_wl=wl_sel,
             frac=frac, cluster=cluster, save_path=save_path)

def run_nested_fit(priors: List, labels: List,
                   bb_params: List, dynesty_params: List,
                   wl_sel: List, path_to_fits: Path,
                   model: Model, pixel_size: int, sampling: int,
                   zero_padding_order: int, method: Optional[str] = "dynamic",
                   frac: Optional[float] = 1e-4,
                   cluster: Optional[bool] = False,
                   save_path: Optional[Path] = "", vis2: Optional[bool] = False,
                   intp: Optional[bool] = True, flux_file: Optional[Path] = None,
                   initial: Optional[List] = None) -> None:
    """Executes an dynesty-fit"""
    data = get_data_for_fit(CompoundModel, pixel_size=pixel_size,
                            sampling=sampling, wl_sel=wl_sel,
                            flux_file=flux_file,
                            zero_padding_order=zero_padding_order,
                            bb_params=bb_params, priors=priors, vis2=vis2,
                            intp=intp, path_to_fits=path_to_fits)

    run_dynesty(dynesty_params, priors, labels, lnlike, data, wl_sel,
                frac=frac, cluster=cluster, method=method, save_path=save_path)


if __name__ == "__main__":
    priors = [[1., 2.], [0, 180], [0.5, 1.], [0, 360], [1., 10.],
              [0., 1.], [0., 1.]]
    initial = get_rndarr_from_bounds(priors, True)
    labels = ["axis ratio", "pos angle", "mod amplitude", "mod angle",
              "inner radius", "tau", "q"]
    bb_params = [1500, 9200, 16, 101.2]
    mcmc_params = [initial, 32, 2500, 5000]
    dynesty_params = [initial, 5000]
    wl_sel = [3.2, 3.45, 3.7]

    path_to_fits = "../../assets/data/SyntheticModels"
    output_path = "../../assets/model_results"
    flux_file = None
    method = "dynamic"

    run_nested_fit(priors, labels, bb_params, dynesty_params,
                   wl_sel, path_to_fits, CompoundModel, 50, 128, 1,
                   method=method)

