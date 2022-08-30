#!/usr/bin/env python3

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
import emcee
import corner
import warnings
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from schwimmbad import MPIPool
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, List, Union, Optional, Callable

from .fourier import FFT
from .baseClasses import Model
from .fitting_utils import chi_sq, lnprob, lnlike

# TODO: Implement global parameter search algorithm (genetic algorithm)
# TODO: Implement optimizer algorithm

def generate_valid_guess(initial: List, priors: List,
                         nwalkers: int, frac: float) -> np.ndarray:
    """Generates a valid guess that is in the bounds of the priors for the
    start of the MCMC-fitting

    Parameters
    ----------
    inital: List
        The inital guess
    priors: List
        The priors that constrain the guess
    nwalkers: int
        The number of walkers to be initialised for

    Returns
    -------
    p0: np.ndarray
        A valid guess for the number of walkers according to the number of
        dimensions
    """
    proposal = np.array(initial)
    prior_dynamic = np.array([np.ptp(i) for i in priors])
    dyn = 1/prior_dynamic

    guess_lst = []
    for i in range(nwalkers):
        guess = proposal + frac*dyn*np.random.normal(proposal, dyn)
        guess_lst.append(guess)

    return np.array(guess_lst, dtype=float)

def plot_corner(sampler: np.ndarray,
                labels: List, wavelength: float,
                save_path: Optional[str] = "") -> None:
    """Plots the corner plot of the posterior spread"""
    samples = sampler.get_chain(flat=True)
    fig = corner.corner(samples, show_titles=True, labels=labels,
                       plot_datapoints=True, quantiles=[0.16, 0.5, 0.84])
    plot_name = f"Corner_plot_{(wavelength):.2f}.png"

    if save_path == "":
        plt.savefig(plot_name)
    else:
        plt.savefig(os.path.join(save_path, plot_name))

def plot_chains(sampler: np.ndarray, theta: List,
                labels: List, wavelength: float,
                save_path: Optional[str] = "") -> None:
    """Plots the chains for debugging to see if and how they converge"""
    fig, axes = plt.subplots(len(theta), figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    ndim = len(theta)

    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number");
    plot_name = f"Chain_plot_{(wavelength):.2f}.png"

    if save_path == "":
        plt.savefig(plot_name)
    else:
        plt.savefig(os.path.join(save_path, plot_name))

def run_mcmc(hyperparams: List, priors: List,
             labels: List, lnprob: Callable, data, plot_wl: float,
             frac: Optional[float] = 1e-4, cluster: Optional[bool] = False,
             synthetic: Optional[bool] = False,
             save_path: Optional[str] = "") -> np.array:
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
    hyperparams: List
    priors: List
    labels: List
    lnprob: Callable
    data: List
    plot_wl: float
    frac: float, optional
    cluster: bool, optional
    synthetic: bool, optional
    save_path: str, optional
    """
    if synthetic:
        try:
            print("Loaded perfect parameters from the synthetic dataset")
            print(np.load("assets/theta.npy"))
        except FileNotFoundError:
            warnings.warn("No 'theta.npy' file could be located!",
                          category=FileNotFoundError)
        finally:
            print("File search done.")

    initial, nwalkers, nburn, niter = hyperparams
    p0 = generate_valid_guess(initial, priors, nwalkers, frac)
    ndim = len(initial)

    print("Inital parameters")
    print(initial)
    print("--------------------------------------------------------------")

    if cluster:
        with MPIPool as pool:
            if not pool.is_master():
                pool.wait()
                sys.exit(0)

            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                            args=data, pool=pool)

            print("Running burn-in...")
            p0, _, _ = sampler.run_mcmc(p0, nburn, progress=True)
            sampler.reset()

            print("--------------------------------------------------------------")
            print("Running production...")
            pos, prob, state = sampler.run_mcmc(p0, niter, progress=True)
            print("--------------------------------------------------------------")

    else:
        with Pool() as pool:
            print(f"Executing MCMC with {cpu_count()} cores.")
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                            args=data, pool=pool)

            print("Running burn-in...")
            p0, _, _ = sampler.run_mcmc(p0, nburn, progress=True)
            sampler.reset()

            print("--------------------------------------------------------------")
            print("Running production...")
            pos, prob, state = sampler.run_mcmc(p0, niter, progress=True)
            print("--------------------------------------------------------------")

    theta_max = (sampler.flatchain)[np.argmax(sampler.flatlnprobability)]

    plot_corner(sampler, labels, plot_wl[0])
    plot_chains(sampler, theta_max, labels, plot_wl[0])
    plot_fit_results(theta_max, *data, hyperparams, labels,
                     plot_wl=plot_wl, save_path=save_path)


if __name__ == "__main__":
    ...

