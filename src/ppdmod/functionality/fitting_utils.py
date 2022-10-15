import os
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from typing import List, Optional
from astropy.units import Quantity

from .combined_model import CombinedModel
from .fourier import FastFourierTransform
from .data_prep import DataHandler
from .plotting_utils import plot_txt, plot_amp_phase_comparison,\
        plot_amp_phase_comparison

def calculate_model(theta: np.ndarray, data: DataHandler):
    data._reformat_theta_to_components(theta)
    model = CombinedModel(data.fixed_params, data.disc_params,
                          data.wavelengths, data.geometric_priors,
                          data.modulation_priors)
    for component in data.model_components:
        model.add_component(component)


    # TODO: Add here the iter data from the DataHandler
    amp_model_data, cphase_model_data = []
    for wavelength in data.wavelengths:
        image = model.eval_flux(wavelength)
        fourier = FastFourierTransform(image, wavelength,
                                       data.pixel_scaling, data.zero_padding_order)



def lnlike(theta: np.ndarray, data: DataHandler) -> float:
    """Takes theta vector and the x, y and the yerr of the theta.
    Returns a number corresponding to how good of a fit the model is to your
    data for a given set of parameters, weighted by the data points.


    Parameters
    ----------
    theta: np.ndarray
        A list of all the parameters that ought to be fitted
    data: DataHandler

    Returns
    -------
    float
        The goodness of the fitted model (will be minimised)
    """

    amp_mod, cphase_mod = model4fit_numerical(theta, model_param_lst,
                                              uv_info_lst, vis_lst)

    amp_chi_sq = chi_sq(amp, sigma2amp, amp_mod)
    cphase_chi_sq = chi_sq(cphase, sigma2cphase, cphase_mod)

    return np.array(-0.5*(amp_chi_sq + cphase_chi_sq), dtype=float)

def lnprior(theta: np.ndarray, priors) -> float:
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

def lnprob(theta: np.ndarray, data: DataHandler) -> np.ndarray:
    """This function runs the lnprior and checks if it returned -np.inf, and
    returns if it does. If not, (all priors are good) it returns the inlike for
    that model (convention is lnprior + lnlike)

    Parameters
    ----------
    theta: List
        A vector that contains all the parameters of the model

    Returns
    -------
    float
        The minimisation value or -np.inf if it fails
    """
    lp = lnprior(theta, data.priors)

    if not np.isfinite(lp):
        return -np.inf

    return lp + lnlike(theta, data)

def chi_sq(real_data: Quantity, sigma_sq: Quantity, model_data: Quantity) -> float:
    """The chi square minimisation"""
    return np.sum(np.log(2*np.pi*sigma_sq) + (real_data-model_data)**2/sigma_sq)

def print_values(data: DataHandler, theta_max: List) -> None:
    """Prints the model's values"""
    print("Best fit corr. fluxes:")
    print(data.model_corr_fluxes)
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

