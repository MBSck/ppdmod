import os

from ppdmod.functionality.mcmc import run_mcmc
from ppdmod.functionality.data_prep import DataHandler
from ppdmod.functionality.utils import make_fixed_params, make_delta_component,\
    make_ring_component

# TODO: Add docstrings to the big overhead functions in this script

# def create_synthetic_dataset(priors: List, bb_params: List, model: Model, fov_size: int,
                             # px_sampling: int, zero_padding_order: Optional[int] = 2,
                             # intp: Optional[bool] = True, fits_file: Optional[Path] = "",
                             # path_to_fits: Optional[Path] = "",
                             # save_path: Optional[str] = "") -> None:
    # """Creates a synthetic dataset by looping over the model for a certain parameter
    # input

    # Parameters
    # ----------
    # priors: List
    # bb_params: List
    # path_to_fits: Path, optional
    # fits_file: Path, optional
    # """
    # theta = get_rndarr_from_bounds(priors)
    # create_model(save_path, model, theta, bb_params, fov_size, px_sampling,
                 # path_to_fits=path_to_fits, fits_file=fits_file)

def run_mcmc_fit() -> None:
    """Executes a mcmc-fit"""
    data_path = "data/hd_142666_jozsef/nband"
    fits_files = ["HD_142666_2022-04-23T03_05_25_N_TARGET_FINALCAL_INT.fits"]
    fits_files = [os.path.join(data_path, file) for file in fits_files]
    flux_files = [None, None]
    save_path = "assets/model_results"
    wavelengths = [12.0]
    data = DataHandler(fits_files, wavelengths, flux_files=flux_files)
    complete_ring = make_ring_component("inner_ring",
                                        [[0., 0.], [0., 0.], [0.1, 6.], [0., 0.]])
    inner_ring = make_ring_component("inner_ring",
                                     [[0., 0.], [0., 0.], [1., 4.], [1., 6.]])
    outer_ring = make_ring_component("outer_ring",
                                     [[0., 0.], [0., 0.], [3., 10.], [0., 0.]])
    delta_component = make_delta_component("star")
    data.add_model_component(delta_component)
    data.add_model_component(complete_ring)
    # data.add_model_component(inner_ring)
    # data.add_model_component(outer_ring)
    data.fixed_params = make_fixed_params(45, 512, 1500, 7900, 140, 19, 1024)
    data.geometric_priors = [[0., 1.], [0, 180]]
    # data.modulation_priors = [[0., 1.], [0, 360]]
    data.disc_priors = [[0., 1.], [0., 1.]]
    data.mcmc = [35, 2, 5, 1e-4]
    data.zero_padding_order = 2
    data.tau_initial = 0.1
    run_mcmc(data, save_path=save_path, cpu_amount=6)


# def run_nested_fit(priors: List, labels: List,
                   # bb_params: List, dynesty_params: List,
                   # wl_sel: List, path_to_fits: Path,
                   # model: Model, pixel_size: int, sampling: int,
                   # zero_padding_order: int, method: Optional[str] = "dynamic",
                   # sampling_method: Optional[str] = "auto",
                   # bounding_method: Optional[str] = "multi",
                   # frac: Optional[float] = 1e-4,
                   # cluster: Optional[bool] = False, synthetic: Optional[bool] = False,
                   # save_path: Optional[Path] = "", vis2: Optional[bool] = False,
                   # intp: Optional[bool] = True, flux_file: Optional[Path] = None,
                   # initial: Optional[List] = None) -> None:
    # """Executes a dynesty-fit"""
    # data = get_data_for_fit(CompoundModel, pixel_size=pixel_size,
                            # sampling=sampling, wl_sel=wl_sel,
                            # flux_file=flux_file,
                            # zero_padding_order=zero_padding_order,
                            # bb_params=bb_params, priors=priors, vis2=vis2,
                            # intp=intp, path_to_fits=path_to_fits)

    # run_dynesty(dynesty_params, priors, labels, lnlike, data, wl_sel,
                # synthetic=synthetic, frac=frac, cluster=cluster,
                # method=method, sampling_method=sampling_method,
                # bounding_method=bounding_method, save_path=save_path)


if __name__ == "__main__":
    run_mcmc_fit()

