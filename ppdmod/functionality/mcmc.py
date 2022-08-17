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
import warnings
import corner
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from schwimmbad import MPIPool
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, List, Union, Optional, Callable

from .fourier import FFT
from .utilities import chi_sq, plot_txt, plot_amp_phase_comparison, lnprob

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

    # NOTE: Switch to np.rand.normal as it gives negative values as well
    guess_lst = []
    for i in range(nwalkers):
        guess = proposal + frac*dyn*np.random.normal(proposal, dyn)
        guess_lst.append(guess)

    return guess_lst

def print_values(realdata: List, datamod: List,
                 theta_max: List, chi_sq_values: List) -> None:
    """Prints the model's values"""
    print("Best fit corr. fluxes:")
    print(datamod[0])
    print("Real corr. fluxes:")
    print(realdata[0])
    print("--------------------------------------------------------------")
    print("Best fit cphase:")
    print(datamod[1])
    print("Real cphase:")
    print(realdata[1])
    print("--------------------------------------------------------------")
    print("Theta max:")
    print(theta_max)
    print("--------------------------------------------------------------")
    print("Chi squared amp:")
    print(chi_sq_values[0])
    print("Chi squared cphase:")
    print(chi_sq_values[1])

def plotter_mcmc(sampler, realdata: List, model_param_lst: List,
                 uv_info_lst: List, vis_lst: List, hyperparams: List,
                 labels: List, plot_wl: float, debug: Optional[bool] = False,
                 plot_px_size: Optional[int] = 2**12,
                 save_path: Optional[str] = "") -> None:
    """Plot the samples to get estimate of the density that has been sampled,
    to test if sampling went well"""
    initial, nwalkers, nburn, niter = hyperparams
    hyperparams_dict = {"nwalkers": nwalkers, "burn-in steps": nburn,
                        "production steps": niter}

    model, pixel_size, sampling, wavelength,\
            zero_padding_order, bb_params, _ = model_param_lst

    amp, cphase = map(lambda x: x[0], map(lambda x: x, realdata[0]))
    amperr, cphaseerr = map(lambda x: np.array(x[0])**2,
                            map(lambda x: x, realdata[1]))

    plot_wl = plot_wl[0]*1e-6
    bb_labels = ["sublimation temperature", "effective temperature",
                 "luminosity of star", "distance to star"]
    bb_params_dict = dict(zip(bb_labels, bb_params))

    uvcoords_lst, u_lst, v_lst, t3phi_uvcoords_lst = map(lambda x: np.array(x),
                                                         uv_info_lst)
    vis, vis2, intp = vis_lst

    if len(u_lst) > 6:
        flux_ind = np.where([i % 6 == 0 for i, o in
                             enumerate(u_lst)])[0].tolist()
        baselines = np.insert(np.sqrt(u_lst**2+v_lst**2), flux_ind, 0.)
    else:
        baselines = np.insert(np.sqrt(u_lst**2+v_lst**2), 0, 0.)

    t3phi_u_lst, t3phi_v_lst = map(lambda x: np.array(x),
                                   map(list, zip(*t3phi_uvcoords_lst)))
    t3phi_baselines = np.sqrt(t3phi_u_lst**2+t3phi_v_lst**2).\
            reshape(len(t3phi_u_lst)//12, 12)
    t3phi_baselines = np.array([np.sort(i)[~3:] for i in t3phi_baselines]).\
            reshape(len(t3phi_u_lst)//12*4)

    theta_max = (sampler.flatchain)[np.argmax(sampler.flatlnprobability)]
    theta_max_dict = dict(zip(labels, theta_max))

    model_cp = model(*bb_params, plot_wl)
    model_flux = model_cp.eval_model(theta_max, pixel_size, sampling)
    fft = FFT(model_flux, plot_wl, pixel_size/sampling,
             zero_padding_order)
    amp_mod, cphase_mod, xycoords = fft.get_uv2fft2(uvcoords_lst,
                                                    t3phi_uvcoords_lst,
                                                    corr_flux=vis, vis2=vis2,
                                                    intp=intp)

    if len(amp_mod) > 6:
        flux_ind = np.where([i % 6 == 0 for i, o in
                             enumerate(amp_mod)])[0].tolist()
        amp_mod = np.insert(amp_mod, flux_ind, np.sum(model_flux))
    else:
        amp_mod = np.insert(amp_mod, 0, np.sum(model_flux))

    _, chi_sq_values = lnlike(theta_max, realdata, model_param_lst,
                              uv_info_lst, vis_lst)
    print_values([amp_mod, cphase_mod], [amp, cphase],
                 theta_max, chi_sq_values)

    if debug:
        plot_corner(sampler, model_cp, labels, plot_wl)
        plot_chains(sampler, model_cp, theta_max, labels, plot_wl)

    fig, axarr = plt.subplots(2, 3, figsize=(20, 10))
    ax, bx, cx = axarr[0].flatten()
    ax2, bx2, cx2 = axarr[1].flatten()

    title_dict = {"Model Fit Parameters": ""}
    text_dict = { "FOV": pixel_size, "npx": sampling,
                 "zero pad order": zero_padding_order, "wavelength": plot_wl,
                 "": "", "blackbody params": "", "---------------------": "",
                 **bb_params_dict, "": "", "best fit values": "",
                 "---------------------": "", **theta_max_dict, "": "",
                 "hyperparams": "", "---------------------": "",
                 **hyperparams_dict}

    plot_txt(ax, title_dict, text_dict, text_font_size=10)
    plot_amp_phase_comparison([[amp, amperr], [amp_mod]],
                              [[cphase, cphaseerr], [cphase_mod]],
                              baselines, t3phi_baselines, [bx, cx])

    fft.plot_amp_phase([fig, ax2, bx2, cx2], corr_flux=True,
                       uvcoords_lst=xycoords)

    plt.tight_layout()
    plot_name = f"{model_cp.name}_model_after_fit_{(plot_wl*1e6):.2f}.png"

    if save_path == "":
        plt.savefig(plot_name)
    else:
        plt.savefig(os.path.join(save_path, plot_name))
    plt.show()

def plot_corner(sampler: np.ndarray, model,
                labels: List, wavelength: float,
                save_path: Optional[str] = "") -> None:
    """Plots the corner plot of the posterior spread"""
    samples = sampler.get_chain(flat=True)
    fig = corner.corner(samples, show_titles=True, labels=labels,
                       plot_datapoints=True, quantiles=[0.16, 0.5, 0.84])
    plot_name = f"{model.name}_corner_plot_{(wavelength*1e6):.2f}.png"

    if save_path == "":
        plt.savefig(plot_name)
    else:
        plt.savefig(os.path.join(save_path, plot_name))

def plot_chains(sampler: np.ndarray, model, theta: List,
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
    plot_name = f"{model.name}_chain_plot_{(wavelength*1e6):.2f}.png"

    if save_path == "":
        plt.savefig(plot_name)
    else:
        plt.savefig(os.path.join(save_path, plot_name))

def run_mcmc(hyperparams: List, priors: List,
             labels: List, lnprob: Callable, data, plot_wl: float,
             frac: Optional[float] = 1e-4, cluster: Optional[bool] = False,
             debug: Optional[bool] = False,
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
    debug: bool, optional
    save_path: str, optional
    """
    initial, nwalkers, nburn, niter = hyperparams
    p0 = generate_valid_guess(initial, priors, nwalkers, frac)
    ndim = len(initial)

    print("Inital parameters")
    print(initial)
    print(p0[0], "p0 Sample")

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
    plotter_mcmc(sampler, *data, hyperparams, labels, plot_wl=plot_wl,
                 debug=debug, save_path=save_path)


if __name__ == "__main__":
    ...

