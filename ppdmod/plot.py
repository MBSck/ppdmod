from typing import Optional, List
from pathlib import Path

import corner
import matplotlib.pyplot as plt
import numpy as np

from .model import Model
from .options import OPTIONS


def plot_corner(sampler: np.ndarray, model: Model,
                save_path: Optional[str] = None) -> None:
    """Plots the corner plot of the posterior spread"""
    samples = sampler.get_chain(flat=True)
    corner.corner(samples, show_titles=True,
                  labels=list(model.params.keys()),
                  plot_datapoints=True,
                  quantiles=[0.16, 0.5, 0.84])

    if save_path is not None:
        plt.savefig(
            Path(save_path) / "corner_plot_.pdf", format="pdf")
    else:
        plt.show()


def plot_chains(theta: List, sampler: np.ndarray,
                save_path: Optional[str] = None) -> None:
    """Plots the chains for debugging to see if and how they converge"""
    _, axes = plt.subplots(len(theta), figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    ndim = len(theta)

    for index in range(ndim):
        axes[index].plot(samples[:, :, index], "k", alpha=0.3)
        axes[index].set_xlim(0, len(samples))
        axes[index].set_ylabel(OPTIONS["model.params"].keys()[index])
        axes[index].yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number")

    if save_path:
        plt.savefig(Path(save_path) / "chain_plot.pdf", format="pdf")
    else:
        plt.show()


# TODO: Finish this.
def plot_amp_phase_comparison(best_fit_total_fluxes,
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

    if data.fit_total_flux:
        max_flux, mod_max_flux = np.max(data.total_fluxes.value),\
                                            np.max(best_fit_total_fluxes.value)
        y_max_flux = max_flux if max_flux > mod_max_flux else mod_max_flux
    else:
        y_max_flux = np.max(best_fit_total_fluxes.value)

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

    # TODO: Add more colors
    color_real_data = ["goldenrod", "darkgoldenrod", "gold"]
    color_fit_data = ["midnightblue", "darkblue", "blue"]

    # TODO: Think about how to get the positional angle and the
    # axis ratio.
    # TODO: Calculate the model here.
    total_fluxes, total_fluxes_err =\
        OPTIONS["data.total_flux"], OPTIONS["data.total_flux_error"]
    corr_fluxes, corr_fluxes_err =\
        OPTIONS["data.correlated_flux"], OPTIONS["data.correlated_flux_error"]
    cphases, cphases_err =\
        OPTIONS["data.closure_phase"], OPTIONS["data.closure_phase_error"]

    for index, (total_flux, total_flux_err, corr_flux,
                corr_flux_err, cphase, cphase_err)\
            in enumerate(
                zip(total_fluxes, total_fluxes_err, corr_fluxes,
                    corr_fluxes_err, cphases, cphases_err)):
        readout = OPTIONS["data.readouts"][index]
        effective_baselines = calculate_effective_baselines(readout.ucoord,
                                                            readout.vcoord,
                                                            axis_ratio,
                                                            pos_angle,
                                                            readout.wavelength)
        longest_baselines = calculate_effective_baselines(readout.u123coord,
                                                          readout.v123coord,
                                                          axis_ratio,
                                                          pos_angle,
                                                          readout.wavelength)
        longest_baselines = np.max(longest_baselines, axis=0)
        ax.errorbar(effective_baselines.value, corr_flux, corr_flux_err,
                    color=color_real_data[index], fmt='o', alpha=0.6)
        ax.scatter(effective_baselines.value, best_fit_corr_fluxes].value,
                   color=color_fit_data[index], marker='X')
        if data.fit_total_flux:
            ax.errorbar(np.array([0]), total_flux, total_flux_err,
                        color=color_real_data[index], fmt='o', alpha=0.6)
            ax.scatter(np.array([0]), best_fit_total_fluxes.value,
                       marker='X', color=color_fit_data[index])
        if data.fit_cphases:
            bx.errorbar(longest_baselines.value,
                        cphase, cphase_err,
                        color=color_real_data[index], fmt='o')
            bx.scatter(longest_baselines.value, best_fit_cphases,
                       color=color_fit_data[index], marker='X')

    ax.set_xlabel(r"$\mathrm{B}_{\mathrm{eff}}/\lambda$ [M$\lambda$]")
    ax.set_ylabel("Correlated fluxes [Jy]")
    ax.set_ylim(y_lim_flux)
    # ax.legend(loc="upper right")

    bx.set_xlabel(r"$\mathrm{B}_{\mathrm{max}}/\lambda$ [M$\lambda$]")
    bx.set_ylabel(fr"Closure Phases [$^\circ$]")
    bx.set_ylim(y_lim_cphase)
    # bx.legend(loc="upper right")


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
