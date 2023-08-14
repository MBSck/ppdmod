from typing import Optional, List
from pathlib import Path

import corner
import matplotlib.pyplot as plt
import numpy as np

from .mcmc import calculate_observables
from .model import Model
from .data import ReadoutFits
from .options import OPTIONS


def plot_corner(sampler: np.ndarray, labels: List[str],
                save_path: Optional[str] = None) -> None:
    """Plots the corner of the posterior spread.

    Parameters
    ----------
    sampler : numpy.ndarray
        The emcee sampler.
    labels : list of str
        The parameter labels.
    save_path : str, optional
        The save path. The default is None.
    """
    samples = sampler.get_chain(flat=True)
    corner.corner(samples, show_titles=True,
                  labels=labels, plot_datapoints=True,
                  quantiles=[0.16, 0.5, 0.84])

    if save_path is not None:
        plt.savefig(
            Path(save_path) / "corner_plot_.pdf", format="pdf")
    else:
        plt.show()


def plot_chains(sampler: np.ndarray, labels: List[str],
                save_path: Optional[str] = None) -> None:
    """Plots the fitter's chains.

    Parameters
    ----------
    sampler : numpy.ndarray
        The emcee sampler.
    labels : list of str
        The parameter labels.
    save_path : str, optional
        The save path. The default is None.
    """
    _, axes = plt.subplots(len(labels), figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()

    for index, label in enumerate(labels):
        axes[index].plot(samples[:, :, index], "k", alpha=0.3)
        axes[index].set_xlim(0, len(samples))
        axes[index].set_ylabel(label)
        axes[index].yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number")

    if save_path:
        plt.savefig(Path(save_path) / "chain_plot.pdf", format="pdf")
    else:
        plt.show()


# TODO: Finish this.
def plot_observed_vs_model(
        image: np.ndarray,
        pixel_size: u.mas,
        wavelength: u.um,
        fits_files: List[ReadoutFits],
        matplot_axes: Optional[List] = None,
        data_to_plot: Optiona[List[str]] = OPTIONS["fit.data"]) -> None:
    """Plots the deviation of a model from real data of an object for
    total flux, visibilities and closure phases.

    Parameters
    ----------
    """
    if matplot_axes is not None:
        ax, bx = matplot_axes
    else:
        fig, axarr = plt.subplots(1, 2, figsize=(10, 5))
        ax, bx = axarr.flatten()

    total_fluxes, total_fluxes_err =\
        OPTIONS["data.total_flux"], OPTIONS["data.total_flux_error"]
    corr_fluxes, corr_fluxes_err =\
        OPTIONS["data.correlated_flux"], OPTIONS["data.correlated_flux_error"]
    cphases, cphases_err =\
        OPTIONS["data.closure_phase"], OPTIONS["data.closure_phase_error"]

    # TODO: Think about how to get the positional angle and the
    # axis ratio.
    # TODO: Calculate the model here.
    for index, (total_flux, total_flux_err, corr_flux,
                corr_flux_err, cphase, cphase_err)\
            in enumerate(
                zip(total_fluxes, total_fluxes_err, corr_fluxes,
                    corr_fluxes_err, cphases, cphases_err)):
        readout = OPTIONS["data.readouts"][index]
        total_flux_model, corr_flux_model, cphase_model =\
            calculate_observables(image, pixel_size, wavelength)

        effective_baselines = calculate_effective_baselines(
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
