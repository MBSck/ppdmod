import os
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Union, Optional, Callable

from .fourier import FFT
from .readout import ReadoutFits
from .baseClasses import Model
from .plotting_utils import plot_txt, plot_amp_phase_comparison,\
        plot_amp_phase_comparison


def get_data_for_fit(model: Model, pixel_size: int, sampling: int,
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
    model: Model
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
                     " default to use 'path_to_fits'", category=ResourceWarning)

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

        for i, wl in enumerate(wl_sel):
            wl_ind = np.where(np.logical_and(wavelength > (wl-average_bin/2),
                                             wavelength < (wl+average_bin/2)))[0].tolist()
            if not wl_ind:
                raise IOError("The wavelength for polychromatic fitting could not be"\
                              " determined... Check the input wavelengths!")

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

            vis, viserr, flux, fluxerr = map(lambda x: np.mean(x, axis=0),
                                             [vis, viserr, flux, fluxerr])

            vis = np.insert(vis, 0, flux)
            viserr = np.insert(viserr, 0, fluxerr) if fluxerr is not None else \
                    np.insert(viserr, 0, flux*0.2)

            vis_lst[i].extend(vis)
            vis_err_lst[i].extend(viserr)

            cphase_lst[i].extend(np.mean(cphase, axis=0))
            cphase_err_lst[i].extend(np.mean(cphaseerr, axis=0))

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

def get_rndarr_from_bounds(bounds: List,
                           centre_rnd: Optional[bool] = False) -> np.ndarray:
    """Initialises a random float/list via a normal distribution from the
    bounds provided

    Parameters
    -----------
    bounds: List
        Bounds list must be nested list(s) containing the bounds of the form
        form [lower_bound, upper_bound]
    centre_rnd: bool, optional
        Get a random number close to the centre of the bound

    Returns
    -------
    float | np.ndarray
    """
    initial = []

    if centre_rnd:
        for lower, upper in bounds:
            if upper == 2:
                initial.append(np.random.normal(1.5, 0.2))
            else:
                initial.append(np.random.normal(upper/2, 0.2))

    else:
        for lower, upper in bounds:
            initial.append(np.random.uniform(lower, upper))

    return np.array(initial, dtype=float)


def model4fit_numerical(theta: np.ndarray, model_param_lst: List,
                        uv_info_lst: List, vis_lst: List) -> np.ndarray:
    """The model image, that is Fourier transformed for the fitting process

    Parameters
    ----------
    theta: np.ndarray
    model_param_lst: List
    uv_info_lst: List
    vis_lst: List

    Returns
    -------
    amp: np.ndarray
        Amplitudes of the model interpolated for the (u, v)-coordinates
    cphases: np.ndarray
        Closure phases of the model interpolated for the (u, v)-coordinates
    """
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

    return np.array(-0.5*(amp_chi_sq + cphase_chi_sq), dtype=float)

def lnprior(theta: np.ndarray, priors: List) -> float:
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

def lnprob(theta: np.ndarray, realdata: List,
           model_param_lst: List, uv_info_lst: List,
           vis_lst: List) -> np.ndarray:
    """This function runs the lnprior and checks if it returned -np.inf, and
    returns if it does. If not, (all priors are good) it returns the inlike for
    that model (convention is lnprior + lnlike)

    Parameters
    ----------
    theta: List
        A vector that contains all the parameters of the model
    realdata: List
    model_param_lst: List
    uv_info_lst: List
    vis_lst: List

    Returns
    -------
    float
        The minimisation value or -np.inf if it fails
    """
    priors = model_param_lst[-1]
    lp = lnprior(theta, priors)

    if not np.isfinite(lp):
        return -np.inf

    return lp + lnlike(theta, realdata, model_param_lst, uv_info_lst, vis_lst)

def chi_sq(real_data: np.ndarray, sigma_sq: np.ndarray,
           model_data: np.ndarray) -> float:
    """The chi square minimisation"""
    return np.sum(np.log(2*np.pi*sigma_sq) + (real_data-model_data)**2/sigma_sq)

def print_values(realdata: List, datamod: List, theta_max: List) -> None:
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

def plot_fit_results(theta_max: List, realdata: List, model_param_lst: List,
                     uv_info_lst: List, vis_lst: List, hyperparams: List,
                     labels: List, plot_wl: List, plot_px_size: Optional[int] = 2**12,
                     save_path: Optional[Path] = "") -> None:
    """Plot the samples to get estimate of the density that has been sampled,
    to test if sampling went well

    Parameters
    ----------
    theta_max: List
    realdata: List
    model_param_lst: List
    uv_info_lst: List
    vis_lst: List
    hyperparams: List
    labels: List
    plot_wl: List
    plot_px_size: int, optional
    save_path: Path, optional
    """
    if len(hyperparams) > 2:
        initial, nwalkers, nburn, niter = hyperparams
        hyperparams_dict = {"nwalkers": nwalkers, "burn-in steps": nburn,
                            "production steps": niter}
    else:
        inital, nlive = hyperparams
        hyperparams_dict = {"nlive": nlive}

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
        flux_ind = np.where([i % 6 == 0 for i, o in enumerate(u_lst)])[0].tolist()
        baselines = np.insert(np.sqrt(u_lst**2+v_lst**2), flux_ind, 0.)
    else:
        baselines = np.insert(np.sqrt(u_lst**2+v_lst**2), 0, 0.)

    t3phi_u_lst, t3phi_v_lst = map(lambda x: np.array(x),
                                   map(list, zip(*t3phi_uvcoords_lst)))
    t3phi_baselines = np.sqrt(t3phi_u_lst**2+t3phi_v_lst**2).\
            reshape(len(t3phi_u_lst)//12, 12)
    t3phi_baselines = np.array([np.sort(i)[~3:] for i in t3phi_baselines]).\
            reshape(len(t3phi_u_lst)//12*4)

    theta_max_dict = dict(zip(labels, theta_max))

    model_cp = model(*bb_params, plot_wl)
    model_flux = model_cp.eval_model(theta_max, pixel_size, sampling)
    fft = FFT(model_flux, plot_wl, pixel_size/sampling,
             zero_padding_order)
    amp_mod, cphase_mod, xycoords = fft.get_uv2fft2(uvcoords_lst, t3phi_uvcoords_lst,
                                                    corr_flux=vis, vis2=vis2, intp=intp)

    if len(amp_mod) > 6:
        flux_ind = np.where([i % 6 == 0 for i, o in enumerate(amp_mod)])[0].tolist()
        amp_mod = np.insert(amp_mod, flux_ind, np.sum(model_flux))
    else:
        amp_mod = np.insert(amp_mod, 0, np.sum(model_flux))

    print_values([amp_mod, cphase_mod], [amp, cphase], theta_max)

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

    fft.plot_amp_phase([fig, ax2, bx2, cx2], corr_flux=True, uvcoords_lst=xycoords)

    plt.tight_layout()
    plot_name = f"{model_cp.name}_model_after_fit_{(plot_wl*1e6):.2f}.png"

    if save_path == "":
        plt.savefig(plot_name)
    else:
        plt.savefig(os.path.join(save_path, plot_name))
    plt.show()


if __name__ == "__main__":
    ...

