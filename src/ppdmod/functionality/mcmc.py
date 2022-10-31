import os
import emcee
import corner
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from pathlib import Path
from typing import Optional
from multiprocessing import Pool, cpu_count

from .data_prep import DataHandler
from .plotting_utils import plot_fit_results, write_data_to_ini
from .fitting_utils import lnprob, calculate_model
from .utils import make_fixed_params, make_delta_component, make_ring_component

np.seterr(divide='ignore')

def generate_valid_guess(data: DataHandler) -> np.ndarray:
    """Generates a valid guess that is in the bounds of the priors for the
    start of the MCMC-fitting

    Parameters
    ----------
    data: DataHandler

    Returns
    -------
    p0: np.ndarray
        A valid guess for the number of walkers according to the number of
        dimensions
    """
    proposal = np.array(data.initial)
    prior_dynamic = np.array([np.ptp(prior) for prior in data.priors])
    dyn = 1/prior_dynamic

    guess_lst = []
    for _ in range(data.mcmc.nwalkers):
        guess = proposal + data.mcmc.frac*dyn*np.random.normal(proposal, dyn)
        guess_lst.append(guess)

    return np.array(guess_lst, dtype=float)

def plot_corner(sampler: np.ndarray, data: DataHandler,
                save_path: Optional[str] = "") -> None:
    """Plots the corner plot of the posterior spread"""
    samples = sampler.get_chain(flat=True)
    corner.corner(samples, show_titles=True, labels=data.labels,
                  plot_datapoints=True, quantiles=[0.16, 0.5, 0.84])
    plot_name = f"Corner-plot_{data.wavelengths[0]}.png"

    if save_path == "":
        plt.savefig(plot_name)
    else:
        plt.savefig(os.path.join(save_path, plot_name))

def plot_chains(sampler: np.ndarray, data: DataHandler,
                save_path: Optional[str] = "") -> None:
    """Plots the chains for debugging to see if and how they converge"""
    fig, axes = plt.subplots(data.mcmc.ndim, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()

    for i in range(data.mcmc.ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(data.labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number");
    plot_name = f"Chain-plot_{data.wavelengths[0]}.png"

    if save_path == "":
        plt.savefig(plot_name)
    else:
        plt.savefig(os.path.join(save_path, plot_name))

def run_mcmc(data: DataHandler,
             cpu_amount: Optional[int] = 6,
             show_plots: Optional[bool] = False,
             save_path: Optional[Path] = None) -> np.array:
    """Runs the emcee Hastings Metropolitan sampler

    The EnsambleSampler recieves the parameters and the args are passed to
    the 'log_prob()' method (an addtional parameter 'a' can be used to
    determine the stepsize, defaults to None).

    The burn-in is first run to explore the parameter space and then the
    walkers settle into the maximum of the density. The state variable is
    then passed to the production run.

    The chain is reset before the production with the state variable being
    passed. 'rstate0' is the state of the internal random number generator

    Parameters
    ----------
    data: DataHandler
    cpu_amount: int, optional
    show_plots: bool, optional
    save_path: str, optional
    """
    p0 = generate_valid_guess(data)
    print("Inital parameters")
    print(data.mcmc.initial)
    print("--------------------------------------------------------------")
    if cpu_amount > cpu_count():
        raise IOError("More cpus specified than available on this node!\n"\
                      f" Cpus specified #{cpu_amount} > Cpus available #{cpu_count()}")

    with Pool(processes=cpu_amount) as pool:
        print(f"Executing MCMC with {cpu_amount} cores.")
        moves = emcee.moves.StretchMove(2.0)
        sampler = emcee.EnsembleSampler(data.mcmc.nwalkers, data.mcmc.ndim,
                                        lnprob, args=[data], pool=pool, moves=moves)

        print("Running burn-in...")
        p0, _, _ = sampler.run_mcmc(p0, data.mcmc.nburn, progress=True)
        sampler.reset()

        print("--------------------------------------------------------------")
        print("Running production...")
        sampler.run_mcmc(p0, data.mcmc.niter, progress=True)
        print("--------------------------------------------------------------")

    data.theta_max = (sampler.flatchain)[np.argmax(sampler.flatlnprobability)]
    best_fit_total_fluxes, best_fit_corr_fluxes, best_fit_cphases, fourier =\
        calculate_model(data.theta_max, data, rfourier=True)

    output_path = f"{datetime.now()}_model_fit"
    if save_path:
        output_path = os.path.join(save_path, output_path)
    os.makedirs(output_path)
    write_data_to_ini(data, best_fit_total_fluxes, best_fit_corr_fluxes,
                      best_fit_cphases, save_path=output_path)
    plot_corner(sampler, data, save_path=output_path)
    plot_chains(sampler, data, save_path=output_path)
    plot_fit_results(best_fit_total_fluxes[0], best_fit_corr_fluxes[0],
                     best_fit_cphases[0], data, fourier, save_path=output_path)

    if show_plots:
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # data_path = "../../../data/hd_142666_jozsef/nband"
    # fits_files = ["HD_142666_2022-04-23T03_05_25_N_TARGET_FINALCAL_INT.fits"]
    # fits_files = [os.path.join(data_path, file) for file in fits_files]
    # flux_files = [None, None]
    fits_files = ["../../../assets/synthetic/2022-10-30 21:28:43.746652_model_synthetic/synthetic_fit_0.fits"]
    save_path = "../../../assets/model_results"
    wavelengths = [12.0]
    data = DataHandler(fits_files, wavelengths)
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
    data.fixed_params = make_fixed_params(30, 512, 1500, 7900, 140, 19, 1024)
    data.geometric_priors = [[0.4, 1.], [0, 180]]
    # data.modulation_priors = [[0., 1.], [0, 360]]
    data.disc_priors = [[0., 1.], [0., 1.]]
    data.lnf_priors = [-10., 10.]
    data.mcmc = [50, 1, 1, 1e-4]
    data.zero_padding_order = 2
    data.tau_initial = 1.
    run_mcmc(data, save_path=save_path, cpu_amount=6, show_plots=True)

