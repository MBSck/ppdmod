from datetime import datetime
from typing import Optional, List
from pathlib import Path

import astropy.units as u
import corner
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib import colormaps as mcm

from .mcmc import calculate_observables
from .model import Model
from .options import OPTIONS
from .utils import calculate_effective_baselines

matplotlib.use('Agg')

def plot_corner(sampler: np.ndarray, labels: List[str],
                discard: Optional[int] = 0,
                savefig: Optional[Path] = None) -> None:
    """Plots the corner of the posterior spread.

    Parameters
    ----------
    sampler : numpy.ndarray
        The emcee sampler.
    labels : list of str
        The parameter labels.
    savefig : pathlib.Path, optional
        The save path. The default is None.
    """
    samples = sampler.get_chain(discard=discard, flat=True)
    corner.corner(samples,
                  show_titles=True,
                  labels=labels, plot_datapoints=True,
                  quantiles=[0.16, 0.5, 0.84])

    if savefig is not None:
        plt.savefig(savefig, format="pdf")
    else:
        plt.show()
    plt.close()


def plot_chains(sampler: np.ndarray, labels: List[str],
                discard: Optional[int] = 0,
                savefig: Optional[Path] = None) -> None:
    """Plots the fitter's chains.

    Parameters
    ----------
    sampler : numpy.ndarray
        The emcee sampler.
    labels : list of str
        The parameter labels.
    savefig : pathlib.Path, optional
        The save path. The default is None.
    """
    samples = sampler.get_chain(discard=discard)
    _, axes = plt.subplots(len(labels), figsize=(10, 7), sharex=True)

    for index, label in enumerate(labels):
        axes[index].plot(samples[:, :, index], "k", alpha=0.3)
        axes[index].set_xlim(0, len(samples))
        axes[index].set_ylabel(label)
        axes[index].yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number")

    if savefig:
        plt.savefig(savefig, format="pdf")
    else:
        plt.show()
    plt.close()


# TODO: Add components to this as well as the parameters in a sub HDULIST.
# As images cubes or sth else?.
def save_model_fits(dim: int, pixel_size: u.mas, distance: u.pc,
                    pos_angle: float, elongation: float,
                    model: Model, wavelength: u.um,
                    savefits: Path, dtype: Optional[np.dtype] = np.float32) -> None:
    """Saves a (.fits)-file of the model with all the information on the parameter space."""
    pixel_size = pixel_size if isinstance(pixel_size, u.Quantity) else pixel_size*u.mas
    wavelength = wavelength if isinstance(wavelength, u.Quantity) else wavelength*u.m
    distance = distance if isinstance(distance, u.Quantity) else distance*u.pc
    pos_angle = pos_angle if isinstance(pos_angle, u.Quantity) else pos_angle*u.deg
    pos_angle_rad = (pos_angle-180*u.deg).to(u.rad).value

    pixel_size_au = pixel_size.to(u.arcsec).value*distance.to(u.pc)
    image = model.calculate_image(dim, pixel_size, wavelength).value.astype(dtype)

    wcs = WCS(naxis=2)
    wcs.wcs.crpix = np.array(image.shape) // 2
    wcs.wcs.cdelt = np.array([pixel_size.value, pixel_size.value])
    wcs.wcs.crval = (0.0, 0.0)
    wcs.wcs.ctype = ("RA---AIR", "DEC--AIR")
    wcs.wcs.cunit = ("mas", "mas")
    wcs.wcs.pc = np.array([[-1, 0], [0, 1]])
    header = wcs.to_header()

    header["BUNIT"] = "Jy", "Unit of original pixel value"
    header["BTYPE"] = "Brightness", "Type of original pixel value"

    header["COMMENT"] = "Best fit model image"

    header["DATAMAX"] = np.max(image), "Quarter star flux (Jy)"
    header["DATE"] = f"{datetime.now()}", "Creation date"
    header["DISTANCE"] = distance.value, "Distance to object (pc)"

    header["EXTEND"] = True, "EXTEND"

    header["LAMBDA"] = np.around(wavelength.value, 2), "Wavelength (microns)"
    header["PA"] = pos_angle.value, "Position angle (deg)"
    header["ELONGRAD"] = np.around(elongation, 2), "Elongation (rad)"
    header["ELONGDEG"] = np.around(elongation*u.rad.to(u.deg), 2), "Elongation (deg)"
    header["OBJECT"] = "HD 142666", "Name of the object"
    # header["LTM1_1"] = np.around(pixel_size_au.value, 5), "Pixel size for x-coordinate (au)"
    # header["LTM2_2"] = np.around(pixel_size_au.value, 5), "Pixel size for y-coordinate (au)"

    hdu = fits.PrimaryHDU(image, header=header)
    hdu.writeto(savefits, overwrite=True)

# TODO: Make inverse plot function from inverse fft.
def plot_model(dim: int, pixel_size: u.mas,
               model: Model, wavelength: u.um,
               zoom: Optional[float] = None,
               savefig: Optional[Path] = None) -> None:
    """Plots the model."""
    pixel_size = pixel_size.value\
        if isinstance(pixel_size, u.Quantity) else pixel_size
    image = model.calculate_image(
        dim, pixel_size, wavelength).value
    disk_max = np.sort(np.unique(image.flatten()))[::-1][1]
    ax_extent = (dim*pixel_size)//2
    plt.imshow(image, vmax=disk_max,
               extent=(-ax_extent, ax_extent,
                       -ax_extent, ax_extent))
    if zoom is not None:
        plt.xlim([-zoom, zoom])
        plt.ylim([-zoom, zoom])
    plt.title(f"Best fit model at {wavelength:.2f}")
    plt.xlabel(r"$\alpha$ (mas)")
    plt.ylabel(r"$\delta$ (mas)")

    if savefig:
        plt.savefig(savefig, format="pdf")
    else:
        plt.show()
    plt.close()


def plot_observed_vs_model(
        model: Model, pixel_size: u.mas, axis_ratio: u.one,
        pos_angle: u.deg, matplot_axes: Optional[List] = None,
        data_to_plot: Optional[List[str]] = OPTIONS["fit.data"],
        savefig: Optional[Path] = None) -> None:
    """Plots the deviation of a model from real data of an object for
    total flux, visibilities and closure phases.

    Parameters
    ----------
    model : Model
        The model.
    pixel_size : astropy.units.mas
        The pixel size.
    axis_ratio : astropy.units.one
        The axis ratio.
    pos_angle : astropy.units.deg
        The position angle.
    matplot_axes : list of matplotlib.axes.Axes, optional
        The matplotlib axes. The default is None.
    data_to_plot : list of str, optional
        The data to plot. The default is OPTIONS["fit.data"].
    savefig : pathlib.Path, optional
        The save path. The default is None.
    """
    wavelengths = OPTIONS["fit.wavelengths"]
    norm = mcolors.Normalize(
            vmin=wavelengths[0].value, vmax=wavelengths[-1].value)
    colormap = mcm.get_cmap("turbo")

    if matplot_axes is not None:
        ax, bx = matplot_axes
    else:
        data_types, nplots = [], 0
        if "vis" in data_to_plot:
            nplots += 1
            data_types.append("vis")

        if "t3phi" in data_to_plot:
            nplots += 1
            data_types.append("t3phi")

    figsize = (12, 5) if nplots == 2 else None
    _, axarr = plt.subplots(1, nplots, figsize=figsize)
    axarr = dict(zip(data_types, axarr.flatten()))

    total_fluxes, total_fluxes_err =\
        OPTIONS["data.total_flux"], OPTIONS["data.total_flux_error"]
    corr_fluxes, corr_fluxes_err =\
        OPTIONS["data.correlated_flux"], OPTIONS["data.correlated_flux_error"]
    cphases, cphases_err =\
        OPTIONS["data.closure_phase"], OPTIONS["data.closure_phase_error"]

    fourier_transforms = {}
    for wavelength in OPTIONS["fit.wavelengths"]:
        fourier_transforms[str(wavelength.value)] =\
            model.calculate_complex_visibility(wavelength)

    for file_index, (total_flux, total_flux_err, corr_flux,
                     corr_flux_err, cphase, cphase_err)\
            in enumerate(
                zip(total_fluxes, total_fluxes_err, corr_fluxes,
                    corr_fluxes_err, cphases, cphases_err)):

        readout = OPTIONS["data.readouts"][file_index]
        effective_baselines = calculate_effective_baselines(
            readout.ucoord, readout.vcoord,
            axis_ratio, pos_angle)[0]
        longest_baselines = calculate_effective_baselines(
            readout.u123coord, readout.v123coord,
            axis_ratio, pos_angle)[0].max(axis=0)

        for wavelength in wavelengths:
            wl_str = str(wavelength.value)
            if wl_str not in corr_flux:
                continue
            total_flux_model, corr_flux_model, cphase_model =\
                calculate_observables(
                    fourier_transforms[wl_str],
                    readout.ucoord, readout.vcoord,
                    readout.u123coord, readout.v123coord,
                    pixel_size, wavelength)

            effective_baselines /= wavelength.value
            longest_baselines /= wavelength.value
            color = colormap(norm(wavelength.value))

            if "vis" in axarr:
                ax = axarr["vis"]
                if "flux" in data_to_plot:
                    ax.errorbar(
                        np.array([0]), total_flux[wl_str],
                        total_flux_err[wl_str], color=color,
                        fmt="o", alpha=0.6)
                    ax.scatter(
                        np.array([0]), total_flux_model,
                        marker="X", color=color)
                if "vis" in data_to_plot:
                    ax.errorbar(
                        effective_baselines.value, corr_flux[wl_str],
                        corr_flux_err[wl_str], color=color, fmt="o", alpha=0.6)
                    ax.scatter(
                        effective_baselines.value, corr_flux_model,
                        color=color, marker="X")
            if "t3phi" in axarr:
                bx = axarr["t3phi"]
                bx.errorbar(
                    longest_baselines.value, cphase[wl_str],
                    cphase_err[wl_str],
                    color=color, fmt="o", alpha=0.6)
                bx.scatter(
                    longest_baselines.value, cphase_model,
                    color=color, marker="X")

    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axarr[data_types[-1]],
                        label="Wavelength (micron)")
    cbar.set_ticks(wavelengths.value)
    cbar.set_ticklabels([f"{wavelength:.0f}" for wavelength in wavelengths.value])

    dot_label = mlines.Line2D([], [], color="k", marker="o",
                              linestyle="None", label="Data", alpha=0.6)
    x_label = mlines.Line2D([], [], color="k", marker="X",
                            linestyle="None", label="Model")
    if "vis" in axarr:
        ax = axarr["vis"]
        ax.set_xlabel(r"$\mathrm{B}_{\mathrm{eff}}/\lambda$ (M$\lambda$)")
        ax.set_ylabel("Correlated fluxes (Jy)")
        # ax.set_ylim(y_lim_flux)
        ax.legend(handles=[dot_label, x_label])

    if "t3phi" in axarr:
        bx = axarr["t3phi"]
        bx.set_xlabel(r"$\mathrm{B}_{\mathrm{max}}/\lambda$ (M$\lambda$)")
        bx.set_ylabel(r"Closure Phases ($^\circ$)")
        # bx.set_ylim(y_lim_cphase)
        bx.legend(handles=[dot_label, x_label])

    if savefig:
        plt.savefig(savefig, format="pdf")
    else:
        plt.show()
    plt.close()
