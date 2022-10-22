import os
import numpy as np
import configparser
import itertools
import matplotlib.pyplot as plt

from pathlib import Path
from astropy.units import Quantity
from typing import Dict, List, Optional

from .utils import IterNamespace
from .data_prep import DataHandler


def plot(image: Quantity) -> None:
    """Plots and image"""
    plt.imshow(image.value)
    plt.show()


def print_results(data: DataHandler, best_fit_total_fluxes,
                  best_fit_corr_fluxes, best_fit_cphases) -> None:
    """Prints the model's values"""
    print("Best fit total fluxes:")
    print(best_fit_total_fluxes)
    print("Best real total fluxes:")
    print(data.total_fluxes[0])
    print("--------------------------------------------------------------")
    print("Best fit correlated fluxes:")
    print(best_fit_corr_fluxes)
    print("Real correlated fluxes:")
    print(data.corr_fluxes[0])
    print("--------------------------------------------------------------")
    print("Best fit cphase:")
    print(best_fit_cphases)
    print("Real cphase:")
    print(data.cphases[0])
    print("--------------------------------------------------------------")
    print("Theta max:")
    print(data.theta_max)


def write_data_to_ini(data: DataHandler, best_fit_total_fluxes,
                      best_fit_corr_fluxes, best_fit_cphases, save_path = "") -> None:
    """Writes the all the data about the model fit into a (.toml)-file"""
    miscellaneous_dict = {"tau": data.tau_initial,
                          "rebin_factor": data.rebin_factor,
                          "wavelengths": data.wavelengths,
                          "uvcoords": data.uv_coords,
                          "uvcoords_closure_phases": data.uv_coords_cphase,
                          "telescope_information": data.telescope_info}
    real_data_dict = {"total_fluxes": data.total_fluxes,
                      "total_fluxes_error": data.total_fluxes_error,
                      "total_fluxes_sigma_squared": data.total_fluxes_sigma_squared,
                      "correlated_fluxes": data.corr_fluxes,
                      "correlated_fluxes_errors": data.corr_fluxes_error,
                      "correlated_fluxes_sigma squared": data.corr_fluxes_sigma_squared,
                      "closure_phases": data.cphases,
                      "closure_phases_errors": data.cphases_error,
                      "closure_phases_sigma_squared": data.cphases_sigma_squared}
    best_fit_data_dict = {"total_fluxes": best_fit_total_fluxes,
                          "correlated_fluxes": best_fit_corr_fluxes,
                          "closure_phases": best_fit_cphases}

    mcmc_dict = data.mcmc.to_string_dict()
    fixed_params_dict = data.fixed_params.to_string_dict()
    best_fit_parameters_dict = IterNamespace(**dict(zip(data.labels, data.theta_max))).to_string_dict()

    config = configparser.ConfigParser()
    config["params.emcee"] = mcmc_dict
    config["params.fixed"] = fixed_params_dict
    config["params.fitted"] = best_fit_parameters_dict

    for component in data.model_components:
        component_dict = component.to_string_dict()
        del component_dict[next(itertools.islice(component_dict, 1, None))]
        config[f"params.components.{component.name}"] = component_dict

    config["params.miscellaneous"] = miscellaneous_dict
    config["data.observed"] = real_data_dict
    config["data.fitted"] = best_fit_data_dict

    file_name = "model_info.ini"
    if save_path is None:
        save_path = file_name
    else:
        save_path = os.path.join(save_path, file_name)

    with open(save_path, "w+") as configfile:
        config.write(configfile)

def plot_fit_results(best_fit_total_fluxes, best_fit_corr_fluxes,
                     best_fit_cphases, data: DataHandler,
                     save_path: Optional[Path] = None) -> None:
    """Plot the samples to get estimate of the density that has been sampled,
    to test if sampling went well

    Parameters
    ----------
    """
    print_results(data, best_fit_total_fluxes, best_fit_corr_fluxes, best_fit_cphases)
    plot_wl = data.wavelengths[0]
    fig, axarr = plt.subplots(2, 3, figsize=(20, 10))
    ax, bx, cx = axarr[0].flatten()
    ax2, bx2, cx2 = axarr[1].flatten()

    plot_amp_phase_comparison(data, best_fit_total_fluxes,
                              best_fit_corr_fluxes, best_fit_cphases,
                              matplot_axes=[bx, cx])
    data.fourier.plot_amp_phase(matplot_axes=[fig, ax2, bx2, cx2],
                                zoom=500, uv_coords=data.uv_coords,
                                uv_coords_cphase=data.uv_coords_cphase)
    plot_name = f"Best-fit-model_{plot_wl}.png"

    if save_path is None:
        plt.savefig(plot_name)
    else:
        plt.savefig(os.path.join(save_path, plot_name))
    plt.tight_layout()
    plt.show()


def plot_amp_phase_comparison(data: DataHandler, best_fit_total_fluxes,
                              best_fit_corr_fluxes, best_fit_cphases,
                              matplot_axes: Optional[List] = []) -> None:
    """Plots the deviation of a model from real data of an object for both
    amplitudes and phases (closure phases)

    Parameters
    ----------
    amp_data: List
        Contains both the model's and the real object's amplitude data and
        errors in the following format [[real_obj, real_err], [model]
    cphase_data: List
        Contains both the model's and the real object's closure phase data and
        errors in the following format [[real_obj, real_err], [model]]
    baselines: List
        The baselines of the amplitudes
    t3phi_baselines: List
        The baselines of the closure phases
    matplot_axes: List, optional
        The axes of matplotlib if this plot is to be embedded in an already
        existing one
    """
    if matplot_axes:
        ax, bx = matplot_axes
    else:
        fig, axarr = plt.subplots(1, 2, figsize=(10, 5))
        ax, bx = axarr.flatten()

    # TODO: Add the total flux to the limit estimation, and check that generally as well
    all_amp = np.concatenate((data.total_fluxes.value[0], best_fit_total_fluxes))
    y_min_amp, y_max_amp = 0, np.max(all_amp)
    y_space_amp = np.sqrt(y_max_amp**2+y_min_amp**2)*0.1
    y_lim_amp = [y_min_amp-y_space_amp, y_max_amp+y_space_amp]

    all_cphase = np.concatenate((data.cphases.value[0], best_fit_cphases))
    y_min_cphase, y_max_cphase = np.min(all_cphase), np.max(all_cphase)
    y_space_cphase = np.sqrt(y_max_cphase**2+y_min_cphase**2)*0.1
    y_lim_cphase = [y_min_cphase-y_space_cphase, y_max_cphase+y_space_cphase]

    count_epochs = data.corr_fluxes[0].shape[0] // 6

    # TODO: Add more colors
    color_real_data = ["goldenrod", "darkgoldenrod", "gold"]
    color_fit_data = ["midnightblue", "darkblue", "blue"]
    for epochs in range(count_epochs):
        if epochs == count_epochs:
            break
        ax.errorbar(data.baselines.value[epochs*6:(epochs+1)*6],
                    data.corr_fluxes.value[0][epochs*6:(epochs+1)*6],
                    data.corr_fluxes_error.value[0][epochs*6:(epochs+1)*6],
                    color=color_real_data[epochs],
                    fmt='o', label="Observed data", alpha=0.6)
        ax.errorbar(np.array([0]), data.total_fluxes[0][epochs],
                    data.total_fluxes_error[0][epochs],
                    color=color_real_data[epochs], fmt='o', alpha=0.6)
        ax.scatter(data.baselines.value[epochs*6:(epochs+1)*6],
                   best_fit_corr_fluxes[epochs*6:(epochs+1)*6],
                   color=color_fit_data[epochs], marker='X', label="Model data")
        ax.scatter(np.array([0]), best_fit_total_fluxes[epochs],
                   marker='X', color=color_fit_data[epochs])
        bx.errorbar(data.longest_baselines.value[epochs*4:(epochs+1)*4],
                    data.cphases.value[0][epochs*4:(epochs+1)*4],
                    data.cphases_error.value[0][epochs*4:(epochs+1)*4],
                    color=color_real_data[epochs], fmt='o',
                    label="Observed data", alpha=0.6)
        bx.scatter(data.longest_baselines.value[epochs*4:(epochs+1)*4],
                   best_fit_cphases[epochs*4:(epochs+1)*4],
                   color=color_fit_data[epochs], marker='X', label="Model data")

    ax.set_xlabel("Baselines [m]")
    ax.set_ylabel("Correlated fluxes [Jy]")
    ax.set_ylim(y_lim_amp)
    ax.legend(loc="upper right")

    bx.set_xlabel("Longest baselines [m]")
    bx.set_ylabel(fr"Closure Phases [$^\circ$]")
    bx.set_ylim(y_lim_cphase)
    bx.legend(loc="upper right")

def plot_txt(ax, title_dict: Dict, text_dict: Dict,
             text_font_size: Optional[int] = 12) -> None:
    """Makes a plot with only text information

    Parameters
    ----------
    ax
        The axis of matplotlib
    input_dict: Dict
        A dict that contains the text as a key and the info as the value
    """
    plot_title = "\n".join([r"$\mathrm{%s}$" % (i) if o == ""\
                            else r"$\mathrm{%s}$: %.2f" % (i.lower(), o)\
                            for i, o in title_dict.items()])
    ax.annotate(plot_title, xy=(0, 1), xytext=(12, -12), va='top',
        xycoords='axes fraction', textcoords='offset points', fontsize=16)
    ax.set_title(plot_title)

    text = "\n".join([r"$\mathrm{%s}$" % (i) if o == ""\
                            else r"$\mathrm{%s}$: %.2f" % (i, o)\
                      for i, o in text_dict.items()])
    ax.annotate(text, xy=(0, 0), xytext=(12, -12), va="bottom",
                xycoords='axes fraction', textcoords='offset points',
                fontsize=text_font_size)

    plt.tight_layout()
    ax.axis('off')


if __name__ == "__main__":
    ...

