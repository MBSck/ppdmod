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

from glob import glob
from pathlib import Path
from schwimmbad import MPIPool
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, List, Union, Optional

from ..models import CompoundModel
from .fourier import FFT
from .readout import ReadoutFits, read_single_dish_txt2np
from .genetic_algorithm import genetic_algorithm, decode
from .utilities import chi_sq, get_rndarr_from_bounds,\
        plot_txt, plot_amp_phase_comparison

# TODO: Make function that randomly assigns starting parameters from priors

# FIXME: The code has some difficulties rescaling for higher pixel numbers
# and does in that case not approximate the right values for the corr_fluxes,
# see pixel_scaling

# TODO: Implement global parameter search algorithm (genetic algorithm)

# TODO: Implement optimizer algorithm

# TODO: Make plots of the model + fitting, that show the visibility curve, and
# two more that show the fit of the visibilities and the closure phases to the
# measured one

# TODO: Make one plot that shows a model of the start parameters and the
# uv-points plotted on top, before fitting starts and options to then change
# them in order to get the best fit (Even multiple times)


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

def get_data(model, pixel_size: int, sampling: int,
             wl_sel: Union[List, float],
             flux_file: Path = None,
             zero_padding_order: Optional[int] = 2,
             bb_params: Optional[List] = [],
             priors: Optional[List] = [],
             vis2: Optional[bool] = False,
             intp: Optional[bool] = False,
             average_bin: Optional[float] = 0.2,
             fits_file: Optional[Path] = [],
             path_to_fits: Optional[Path] = "") -> List:
    """Fetches the required info from the '.fits'-files and then returns a
    tuple containing it.

    If no wavelength index ('wl_ind') is provided, it fetches the data for all
    the wavelengths (polychromatic)

    If a few wavelength indices are provided, it will fetch them and average
    them with an area around them.

    The data can also be interpolated downwards, to have less gridpoints based
    on the resolution of the problem.

    Parameters
    ----------
    model
        The model that is to be calculated
    pixel_size: int
        The size of the FOV, that is used
    sampling: int
        The amount of pixels used in the model image
    wl_sel: List | float
        Picks the wavelengths to be fitted to
    flux_file: Path, optional
        An additional '.fits'-file that contains the flux of the object
    zero_padding_order: int, optional
        The order of the zero padding
    bb_params: List
        The blackbody parameters that are used for Planck's law
    priors: List, optional
        The priors that set the bounds for the fitting algorithm
    intp: bool, optional
        Determines if it interpolates or rounds to the nearest pixel
    average_bin: float, optional
        The area around the wl_ind's return value around which is averaged.
        Input is in microns
    intp_down: bool, optional
        Interpolates the wavelength grid downwards to fit a certain number of
        parameters
    fits_file: Path, optional
        The path to a (.fits)-file
    path_to_fits: Path, optional
        The path from which the (.fits)-files are to be accessed. Either
        'fits_files' or 'path_to_fits' must be given

    Returns
    -------
    tuple
    """
    if path_to_fits:
        fits_files = glob(os.path.join(path_to_fits, "*.fits"))
    elif fits_file:
        fits_files = [fits_file]
    else:
        raise IOError("Either 'path_to_fits' or 'fits_file' must be set!")

    if path_to_fits and fits_file:
        warnings.warn("Both 'path_to_fits' and 'fits_file' are set, this will"\
                     " default to use 'path_to_fits', category=ResourceWarning")

    vis_lst, vis_err_lst = [[] for _ in wl_sel], [[] for _ in wl_sel]
    cphase_lst, cphase_err_lst = [[] for _ in wl_sel], [[] for _ in wl_sel]
    flux_lst, flux_err_lst = [[] for _ in wl_sel], [[] for _ in wl_sel]
    u_lst, v_lst, uvcoords_lst = [], [], []
    t3phi_uvcoords_lst = [[[] for _ in range(3)] for _ in range(2)]

    wl_sel = [i*1e-6 for i in wl_sel]
    average_bin *= 1e-6

    for fits_file in fits_files:
        readout = ReadoutFits(fits_file)
        wavelength = readout.get_wl()

        for i, o in enumerate(wl_sel):
            wl_ind = np.where(np.logical_and(wavelength > (o-average_bin/2),
                                             wavelength < (o+average_bin/2)))[0].tolist()

            if vis2:
                vis, viserr = map(list, zip(*[readout.get_vis24wl(x) for x in wl_ind]))
            else:
                vis, viserr = map(list, zip(*[readout.get_vis4wl(x) for x in wl_ind]))

            cphase, cphaseerr = map(list, zip(*[readout.get_t3phi4wl(x) for x in wl_ind]))

            if flux_file:
                # FIXME: Check if there is real fluxerr -> If yes, add it
                # FIXME: Check if this still works with the polychromatic approach
                flux = read_single_dish_txt2np(flux_file, wavelength)[wavelength[wl_ind]]
                fluxerr = None
            else:
                flux, fluxerr = map(list, zip(*[readout.get_flux4wl(x) for x in wl_ind]))

            # NOTE: Fluxerr is just 20% of flux, if no fluxerr is given
            vis, viserr, flux, fluxerr = map(lambda x: np.mean(x, axis=0),
                                             [vis, viserr, flux, fluxerr])

            vis = np.insert(vis, 0, flux)
            viserr = np.insert(viserr, 0, fluxerr) if fluxerr is not None else \
                    np.insert(viserr, 0, flux*0.2)

            vis_lst[i].extend(vis)
            vis_err_lst[i].extend(viserr)

            # FIXME: cannot mean over all cphase as more than one fits file
            cphase_lst[i].extend(np.mean(cphase, axis=0))
            cphase_err_lst[i].extend(np.mean(cphaseerr, axis=0))

            # FIXME: Check how the flux is the calculated vs the model
            flux_lst[i].append(flux)
            flux_err_lst[i].append(fluxerr)

        u, v = readout.get_split_uvcoords()
        u_lst.extend(u)
        v_lst.extend(v)
        uvcoords_lst.extend(readout.get_uvcoords())

        for i, o in enumerate(t3phi_uvcoords_lst):
            for j, l in enumerate(o):
                l.extend(readout.get_t3phi_uvcoords()[i][j])

    t3phi_uvcoords_lst_reformatted = []
    for x, y in zip(*t3phi_uvcoords_lst):
        t3phi_uvcoords_lst_reformatted.extend(list(zip(x, y)))

    data = ([vis_lst, cphase_lst], [vis_err_lst, cphase_err_lst])
    uv_info_lst = [uvcoords_lst, u_lst, v_lst, t3phi_uvcoords_lst_reformatted]
    model_param_lst = [model, pixel_size, sampling, wl_sel,
                       zero_padding_order, bb_params, priors]
    bool_lst = [not vis2, vis2, intp]

    return [data, model_param_lst, uv_info_lst, bool_lst]

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

def model4fit_numerical(theta: np.ndarray, model_param_lst,
                        uv_info_lst, vis_lst) -> np.ndarray:
    """The model image, that is Fourier transformed for the fitting process"""
    model, pixel_size, sampling, wavelength,\
            zero_padding_order, bb_params, _ = model_param_lst
    uvcoords, u, v, t3phi_uvcoords = uv_info_lst
    vis, vis2, intp = vis_lst

    amp_lst, cphase_lst = [], []

    for i in wavelength:
        model_base = model(*bb_params, i)
        model_flux = model_base.eval_model(theta, pixel_size, sampling)
        fft = FFT(model_flux, i, model_base.pixel_scale,
                 zero_padding_order)
        amp, cphase, xycoords = fft.get_uv2fft2(uvcoords, t3phi_uvcoords,
                                               corr_flux=vis, vis2=vis2,
                                               intp=intp)
        if len(amp) > 6:
            flux_ind = np.where([i % 6 == 0 for i, o in
                                 enumerate(amp)])[0].tolist()
            amp = np.insert(amp, flux_ind, np.sum(model_flux))
        else:
            amp = np.insert(amp, 0, np.sum(model_flux))

        amp_lst.append(amp)
        cphase_lst.append(cphase)

    return np.array(amp_lst), np.array(cphase_lst)

def lnlike(theta: np.ndarray, realdata: List,
           model_param_lst: List, uv_info_lst: List,
           vis_lst: List) -> float:
    """Takes theta vector and the x, y and the yerr of the theta.
    Returns a number corresponding to how good of a fit the model is to your
    data for a given set of parameters, weighted by the data points.


    Parameters
    ----------
    theta: np.ndarray
        A list of all the parameters that ought to be fitted
    realdata: List
    model_param_lst: List
    uv_info_lst: List
    vis_lst: List

    Returns
    -------
    float
        The goodness of the fitted model (will be minimised)
    """
    amp, cphase = map(lambda x: np.array(x), realdata[0])
    amperr, cphaseerr = map(lambda x: np.array(x), realdata[1])

    if amperr.shape != (7,):
        sigma2amp = np.array([[x**2 for x in i] for i in amperr])
        sigma2cphase = np.array([[x**2 for x in i] for i in cphaseerr])
    else:
        sigma2amp, sigma2cphase= map(lambda x: x**2, [amperr, cphaseerr])

    amp_mod, cphase_mod = model4fit_numerical(theta, model_param_lst,
                                              uv_info_lst, vis_lst)

    amp_chi_sq = chi_sq(amp, sigma2amp, amp_mod)
    cphase_chi_sq = chi_sq(cphase, sigma2cphase, cphase_mod)

    return -0.5*(amp_chi_sq + cphase_chi_sq), [amp_chi_sq, cphase_chi_sq]

def lnprior(theta: np.ndarray, priors: List):
    """Checks if all variables are within their priors (as well as
    determining them setting the same).

    If all priors are satisfied it needs to return '0.0' and if not '-np.inf'
    This function checks for an unspecified amount of flat priors. If upper
    bound is 'None' then no upper bound is given

    Parameters
    ----------
    theta: np.ndarray
        A list of all the parameters that ought to be fitted
    priors: List
        A list containing all the prior's bounds

    Returns
    -------
    float
        Return-code 0.0 for within bounds and -np.inf for out of bound
        priors
    """
    check_conditons = []

    for i, o in enumerate(priors):
        if o[1] is None:
            if o[0] < theta[i]:
                check_conditons.append(True)
            else:
                check_conditons.append(False)
        else:
            if o[0] < theta[i] < o[1]:
                check_conditons.append(True)
            else:
                check_conditons.append(False)

    return 0.0 if all(check_conditons) else -np.inf

def lnprob(theta: np.ndarray, realdata,
           model_param_lst, uv_info_lst,
           vis_lst) -> np.ndarray:
    """This function runs the lnprior and checks if it returned -np.inf, and
    returns if it does. If not, (all priors are good) it returns the inlike for
    that model (convention is lnprior + lnlike)

    Parameters
    ----------
    theta: List
        A vector that contains all the parameters of the model

    Returns
    -------
    """
    priors = model_param_lst[-1]
    lp = lnprior(theta, priors)

    if not np.isfinite(lp):
        return -np.inf

    return lp +\
            lnlike(theta, realdata, model_param_lst, uv_info_lst, vis_lst)[0]

def do_mcmc(hyperparams: List, priors,
            labels, lnprob, data, plot_wl: float,
            frac: Optional[float] = 1e-4,
            cluster: Optional[bool] = False,
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
    lnprob
    data: List
    plot_wl: float
    frac: float, optional
    cluster: bool, optional
    debug: bool, optional
    save_path: str, optional
    """
    initial, nwalkers, nburn, niter = hyperparams
    p0 = generate_valid_guess(initial, priors, nwalkers, frac)
    print(initial, "Inital params")
    print(p0[0], "p0 Sample")
    ndim = len(initial)

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
            ncores = cpu_count()
            print(f"Executing MCMC with {ncores} cores.")
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
    priors = [[1., 2.], [0, 180], [0.5, 1.], [0, 360], [1., 10.],
              [0., 1.], [0., 1.]]
    initial = get_rndarr_from_bounds(priors, True)
    labels = ["axis ratio", "pos angle", "mod amplitude", "mod angle",
              "inner radius", "tau", "q"]
    bb_params = [1500, 9200, 16, 101.2]
    mcmc_params = [initial, 32, 2500, 5000]
    wl_sel = [3.2, 3.45, 3.7]

    path_to_fits = "../../assets/data/SyntheticModels"
    output_path = "../../assets/model_results"
    flux_file = None

    data = get_data(CompoundModel, pixel_size=50,
                    sampling=128, wl_sel=wl_sel,
                    flux_file=flux_file,
                    zero_padding_order=1, bb_params=bb_params,
                    priors=priors, vis2=False, intp=True,
                    path_to_fits=path_to_fits)

    do_mcmc(mcmc_params, priors, labels, lnprob, data, plot_wl=wl_sel,
            frac=1e-4, cluster=False, debug=True)

