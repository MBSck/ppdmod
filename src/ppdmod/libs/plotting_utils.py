import os
import numpy as np
import configparser
import itertools
import matplotlib.pyplot as plt
import astropy.units as u

from pathlib import Path
from astropy.units import Quantity
from typing import Dict, List, Optional

from .utils import IterNamespace, calculate_effective_baselines
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
    print("Real total fluxes:")
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
    print("Closure phases have not been fitted!")
    print("--------------------------------------------------------------")
    print("Theta max:")
    print(data.theta_max)


def write_data_to_ini(data: DataHandler, best_fit_total_fluxes,
                      best_fit_corr_fluxes, best_fit_cphases,
                      save_path: Optional[Path] = "") -> None:
    """Writes the all the data about the model fit into a (.toml)-file"""
    miscellaneous_dict = {"fits_files": data.fits_files,
                          "tau": data.tau_initial, "rebin_factor": data.rebin_factor,
                          "wavelengths": data.wavelengths,
                          "uvcoords": data.uv_coords,
                          "uvcoords_closure_phases": data.uv_coords_cphase,
                          "telescope_information": data.telescope_info}
    real_data_dict = {"total_fluxes": data.total_fluxes,
                      "total_fluxes_error": data.total_fluxes_error,
                      "correlated_fluxes": data.corr_fluxes,
                      "correlated_fluxes_errors": data.corr_fluxes_error,
                      "closure_phases": data.cphases,
                      "closure_phases_errors": data.cphases_error}
    best_fit_data_dict = {"total_fluxes": best_fit_total_fluxes,
                          "correlated_fluxes": best_fit_corr_fluxes,
                          "closure_phases": best_fit_cphases}

    mcmc_dict = {} if data.mcmc is None else data.mcmc.to_string_dict()
    dynesty_dict = {} if data.dynesty is None else data.dynesty.to_string_dict()
    fixed_params_dict = data.fixed_params.to_string_dict()
    best_fit_parameters_dict = IterNamespace(**dict(zip(data.labels,
                                                        data.theta_max))).to_string_dict()

    config = configparser.ConfigParser()
    config["params.fixed"] = fixed_params_dict
    config["params.fitted"] = best_fit_parameters_dict

    config["params.fit.emcee"] = mcmc_dict
    config["params.fit.dynesty"] = dynesty_dict

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
                     best_fit_cphases, data: DataHandler, fourier,
                     save_path: Optional[Path] = None) -> None:
    """Plot the samples to get estimate of the density that has been sampled,
    to test if sampling went well

    Parameters
    ----------
    """
    print_results(data, best_fit_total_fluxes,
                  best_fit_corr_fluxes, best_fit_cphases)
    fig, axarr = plt.subplots(2, 3, figsize=(20, 10))
    ax, bx, cx = axarr[0].flatten()
    ax2, bx2, cx2 = axarr[1].flatten()

    plot_amp_phase_comparison(data, best_fit_total_fluxes,
                              best_fit_corr_fluxes, best_fit_cphases,
                              matplot_axes=[bx, cx])
    fourier.plot_amp_phase(matplot_axes=[fig, ax2, bx2, cx2],
                           zoom=500, uv_coords=data.uv_coords,
                           uv_coords_cphase=data.uv_coords_cphase)
    plot_name = f"Best-fit-model_{data.wavelengths[0]}.png"

    if save_path is None:
        plt.savefig(plot_name)
    else:
        plt.savefig(os.path.join(save_path, plot_name))


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

    max_flux, mod_max_flux = np.max(data.total_fluxes.value),\
                                        np.max(best_fit_total_fluxes.value)
    y_max_flux = max_flux if max_flux > mod_max_flux else mod_max_flux
    y_space_flux = np.sqrt(y_max_flux**2)*0.1
    y_lim_flux = [0, y_max_flux+y_space_flux]

    min_cphases, max_cphases = np.min(data.cphases.value), np.max(data.cphases.value)
    min_mod_cphases, max_mod_cphases = np.min(best_fit_cphases.value),\
            np.max(best_fit_cphases.value)
    y_min_cphase = min_cphases if min_cphases < min_mod_cphases\
            else min_mod_cphases
    y_max_cphase = max_cphases if max_cphases > max_mod_cphases\
            else max_mod_cphases
    y_space_cphase = np.sqrt(y_max_cphase**2+y_min_cphase**2)*0.1
    y_lim_cphase = [y_min_cphase-y_space_cphase, y_max_cphase+y_space_cphase]

    count_epochs = data.corr_fluxes[0].shape[0] // 6

    # TODO: Add more colors
    color_real_data = ["goldenrod", "darkgoldenrod", "gold"]
    color_fit_data = ["midnightblue", "darkblue", "blue"]
    axis_ratio = data.theta_max[0]*u.dimensionless_unscaled
    pos_angle = (data.theta_max[1]*u.deg).to(u.rad)

    for epochs in range(count_epochs):
        for index, corr_fluxes in enumerate(data.corr_fluxes):
            effective_baselines = calculate_effective_baselines(data.uv_coords,
                                                                axis_ratio,
                                                                pos_angle,
                                                                data.wavelengths[index])
            longest_baselines = calculate_effective_baselines(data.uv_coords_cphase,
                                                              axis_ratio,
                                                              pos_angle,
                                                              data.wavelengths[index])
            longest_baselines = np.max(longest_baselines, axis=0)
            ax.errorbar(effective_baselines.value[epochs*6:(epochs+1)*6],
                        corr_fluxes.value[epochs*6:(epochs+1)*6],
                        data.corr_fluxes_error[index].value[epochs*6:(epochs+1)*6],
                        color=color_real_data[epochs],
                        fmt='o', alpha=0.6)
            ax.scatter(effective_baselines.value[epochs*6:(epochs+1)*6],
                       best_fit_corr_fluxes[index].value[epochs*6:(epochs+1)*6],
                       color=color_fit_data[epochs], marker='X')
            if data.fit_total_flux:
                ax.errorbar(np.array([0]), data.total_fluxes[index][epochs],
                            data.total_fluxes_error[index][epochs],
                            color=color_real_data[epochs], fmt='o', alpha=0.6)
                ax.scatter(np.array([0]), best_fit_total_fluxes.value[epochs],
                           marker='X', color=color_fit_data[epochs])
            if data.fit_cphases:
                bx.errorbar(longest_baselines.value[epochs*4:(epochs+1)*4],
                            data.cphases[index].value[epochs*4:(epochs+1)*4],
                            data.cphases_error[index].value[epochs*4:(epochs+1)*4],
                            color=color_real_data[epochs], fmt='o')
                bx.scatter(longest_baselines.value[epochs*4:(epochs+1)*4],
                           best_fit_cphases[index].value[epochs*4:(epochs+1)*4],
                           color=color_fit_data[epochs], marker='X')

    ax.set_xlabel(r"$\mathrm{B}_{\mathrm{eff}}/\lambda$ [M$\lambda$]")
    ax.set_ylabel("Correlated fluxes [Jy]")
    ax.set_ylim(y_lim_flux)
    # ax.legend(loc="upper right")

    bx.set_xlabel(r"$\mathrm{B}_{\mathrm{max}}/\lambda$ [M$\lambda$]")
    bx.set_ylabel(fr"Closure Phases [$^\circ$]")
    bx.set_ylim(y_lim_cphase)
    # bx.legend(loc="upper right")


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

