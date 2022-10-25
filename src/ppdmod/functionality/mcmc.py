"""Test file for a 2D-Gaussian PPD model, that is fit with MCMC; The emcee
package

...

Initial sets the theta

>>> initial = np.array([1.5, 135, 1., 1., 100., 3., 0.01, 0.7])
>>> priors = [[1., 2.], [0, 180], [0., 2.], [0., 2.], [0., 180.], [1., 10.],
              [0., 1.], [0., 1.]]
>>> labels = ["AXIS_RATIO", "P_A", "C_AMP", "S_AMP", "MOD_ANGLE", "R_INNER",
              "TAU", "Q"]
>>> bb_params = [1500, 7900, 19, 140]

File to read data from

>>> f = "../../assets/Final_CAL.fits"
>>> out_path = "../../assets"

sws is for L-band flux; timmi2 for the N-band flux

>>> flux_file = "../../assets/HD_142666_timmi2.txt"

Set the data, the wavelength has to be the fourth argument [3]

>>> data = set_data(fits_file=f, flux_file=flux_file, pixel_size=100,
                    sampling=128, wl_ind=38, zero_padding_order=3, vis2=False)

Set the mcmc parameters and the data to be fitted.

>>> mc_params = set_mc_params(initial=initial, nwalkers=50, niter_burn=100,
                              niter=250)

This calls the MCMC fitting

>>> fitting = ModelFitting(CompoundModel, data, mc_params, priors, labels,
                           numerical=True, vis=True, modulation=True,
                           bb_params=bb_params, out_path=out_path)
>>> fitting.pipeline()

"""
import os
import time
import emcee
import corner
import numpy as np
import matplotlib.pyplot as plt

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
    print(proposal)
    prior_dynamic = np.array([np.ptp(prior) for prior in data.priors])
    dyn = 1/prior_dynamic
    print(prior_dynamic)

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

    output_path = f"{time.time()}_results_fit"
    if save_path:
        output_path = os.path.join(save_path, output_path)
    os.makedirs(output_path)
    write_data_to_ini(data, best_fit_total_fluxes, best_fit_corr_fluxes,
                      best_fit_cphases, save_path=output_path)
    plot_corner(sampler, data, save_path=output_path)
    plot_chains(sampler, data, save_path=output_path)
    plot_fit_results(best_fit_total_fluxes[0], best_fit_corr_fluxes[0],
                     best_fit_cphases[0], data, fourier, save_path=output_path)


if __name__ == "__main__":
    data_path = "../../../data/hd_142666_jozsef/nband"
    fits_files = ["HD_142666_2022-04-23T03_05_25_N_TARGET_FINALCAL_INT.fits"]
    fits_files = [os.path.join(data_path, file) for file in fits_files]
    flux_files = [None, None]
    save_path = "../../../assets/model_results"
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

