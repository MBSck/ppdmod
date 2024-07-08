import re
from typing import Optional, Dict, List
from pathlib import Path

import astropy.units as u
import astropy.constants as const
import corner
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from dynesty import plotting as dyplot
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from .component import Component
from .fitting import compute_observables
from .options import OPTIONS, get_colormap
from .utils import compute_effective_baselines, restrict_phase, \
        set_legend_color, set_axes_color

matplotlib.use("Agg")


def plot_component_mosaic(components: List[Component], dim: int,
                          pixel_size: u.mas, wavelengths: u.um,
                          norm: Optional[float] = 0.5,
                          zoom: Optional[float] = None,
                          cols: Optional[int] = 4,
                          cmap: Optional[str] = "inferno",
                          savefig: Optional[Path] = None) -> None:
    """Plots a mosaic of components for the different wavelengths."""
    num_plots = np.array(wavelengths).size
    rows = int(np.ceil(num_plots/cols))
    fig, axes = plt.subplots(
        rows, cols, figsize=(5, 5), facecolor=OPTIONS.plot.color.background)

    for ax, wavelength in zip(axes.flatten(), wavelengths):
        plot_components(components, dim, pixel_size, wavelength,
                        norm=norm, zoom=zoom, ax=ax, cmap=cmap)

    [fig.delaxes(ax) for ax in axes.flatten()[num_plots:]]
    plt.tight_layout()

    if savefig is not None:
        plt.savefig(savefig, format=Path(savefig).suffix[1:])
    else:
        plt.show()
    plt.close()


# TODO: Add here the text in the top left corner for the wavelength
# and a scale for the au
def plot_components(components: List[Component], dim: int,
                    pixel_size: u.mas, wavelength: u.um,
                    norm: Optional[float] = 0.5,
                    save_as_fits: Optional[bool] = False,
                    zoom: Optional[float] = None,
                    ax: Optional[bool] = False,
                    cmap: Optional[str] = "inferno",
                    savefig: Optional[Path] = None) -> None:
    """Plots a component."""
    components = [components] if not isinstance(components, list) else components
    image = sum([comp.compute_image(dim, pixel_size, wavelength) for comp in components])
    extent = [sign * dim * pixel_size / 2 for sign in [-1, 1, 1, -1]]
    if save_as_fits:
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = (dim // 2, dim // 2)
        wcs.wcs.cdelt = (-pixel_size*u.mas.to(u.deg), -pixel_size*u.mas.to(u.deg))
        wcs.wcs.crval = (0.0, 0.0)
        hdu = fits.HDUList([fits.PrimaryHDU(image, header=wcs.to_header())])
        hdu.writeto(savefig, overwrite=True)
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1)
        ax.imshow(image[0], extent=extent,
                  norm=mcolors.PowerNorm(gamma=norm), cmap=cmap)
        ax.text(-22, 22, fr"$\lambda$ = {wavelength} " +r"$\mathrm{\mu}$m", fontsize=12, color="white")
        ax.set_xlim([zoom, -zoom])
        ax.set_ylim([-zoom, zoom])

        ax.set_xlabel(r"$\alpha$ (mas)")
        ax.set_ylabel(r"$\delta$ (mas)")

        if savefig is not None:
            plt.savefig(savefig, format=Path(savefig).suffix[1:])


def plot_corner(sampler: np.ndarray, labels: List[str],
                units: Optional[List[str]] = None,
                discard: Optional[int] = 0,
                savefig: Optional[Path] = None,
                **kwargs) -> None:
    """Plots the corner of the posterior spread.

    Parameters
    ----------
    sampler : numpy.ndarray
        The emcee sampler.
    labels : list of str
        The parameter labels.
    units : list of str, optional
    discard : int, optional
    savefig : pathlib.Path, optional
        The save path. The default is None.
    """
    if units is not None:
        labels = [f"{label} [{unit}]" for label, unit in zip(labels, units)]
    quantiles = [x/100 for x in OPTIONS.fit.quantiles]
    if OPTIONS.fit.method == "emcee":
        samples = sampler.get_chain(discard=discard, flat=True)
        corner.corner(samples, show_titles=True,
                      labels=labels, plot_datapoints=True,
                      quantiles=quantiles, title_kwargs={"fontsize": 12})
    else:
        results = sampler.results
        dyplot.cornerplot(results, color='blue',
                          truths=np.zeros(len(labels)),
                          labels=labels, truth_color='black',
                          show_titles=True, max_n_ticks=3,
                          title_quantiles=quantiles,
                          quantiles=quantiles)

    if savefig is not None:
        plt.savefig(savefig, format="pdf")
    else:
        plt.show()
    plt.close()


def plot_chains(sampler: np.ndarray, labels: List[str],
                units: Optional[List[str]] = None,
                discard: Optional[int] = 0,
                savefig: Optional[Path] = None,
                **kwargs) -> None:
    """Plots the fitter's chains.

    Parameters
    ----------
    sampler : numpy.ndarray
        The emcee sampler.
    labels : list of str
        The parameter labels.
    units : list of str, optional
    discard : int, optional
    savefig : pathlib.Path, optional
        The save path. The default is None.
    """
    if units is not None:
        labels = [f"{label} [{unit}]" for label, unit in zip(labels, units)]

    quantiles = [x/100 for x in OPTIONS.fit.quantiles]
    if OPTIONS.fit.method == "emcee":
        samples = sampler.get_chain(discard=discard)
        _, axes = plt.subplots(len(labels), figsize=(10, 7), sharex=True)

        for index, label in enumerate(labels):
            axes[index].plot(samples[:, :, index], "k", alpha=0.3)
            axes[index].set_xlim(0, len(samples))
            axes[index].set_ylabel(label)
            axes[index].yaxis.set_label_coords(-0.1, 0.5)
        axes[-1].set_xlabel("step number")
    else:
        results = sampler.results
        dyplot.traceplot(results, labels=labels,
                         truths=np.zeros(len(labels)),
                         quantiles=quantiles,
                         truth_color="black", show_titles=True,
                         trace_cmap="viridis", connect=True,
                         connect_highlight=range(5))

    if savefig:
        plt.savefig(savefig, format="pdf")
    else:
        plt.show()
    plt.close()


def plot_model(fits_file: Path, data_type: Optional[str] = "image",
               wavelength: Optional[float] = None,
               pixel_size: Optional[float] = None,
               factor: Optional[float] = None,
               zoom: Optional[int] = 30,
               colormap: Optional[str] = OPTIONS.plot.color.colormap,
               title: Optional[str] = None,
               savefig: Optional[Path] = None) -> None:
    """Plots the model information stored in the (.fits)-file.

    Parameters
    ----------
    fits_file : pathlib.Path
    data_type : str, optional
    title : float, optional
    savefig : pathlib.Path
    """
    _ = plt.figure(facecolor=OPTIONS.plot.color.background)
    ax = plt.axes(facecolor=OPTIONS.plot.color.background)
    set_axes_color(ax, OPTIONS.plot.color.background)
    colormap = get_colormap(colormap)
    with fits.open(fits_file) as hdul:
        if data_type == "image":
            if pixel_size is None or wavelength is None:
                raise ValueError("Pixel_size and Wavelength must be specified"
                                 "for image plotting.")
            index = np.where(hdul[1].data["wavelength"] == wavelength)
            image = np.log(1+hdul[0].data[index][0])
            extent = [-image.shape[0]*pixel_size/2, image.shape[0]*pixel_size/2,
                      -image.shape[1]*pixel_size/2, image.shape[1]*pixel_size/2]
            plt.imshow(image, extent=extent)
            plt.ylim([-zoom, zoom])
            plt.xlim([-zoom, zoom])
            plt.xlabel(r"$\alpha$ (mas)")
            plt.ylabel(r"$\delta$ (mas)")
        else:
            wavelengths = hdul[1].data["wavelength"]
            radius = hdul["FULL_DISK"].data["radius"][0]
            if data_type == "temperature":
                plt.plot(radius, hdul["FULL_DISK"].data["temperature"][0])
                plt.ylabel(r"Temperature (K)")
                plt.xlabel(r"Radius (mas)")

            elif data_type == "brightness":
                for wavelength, data in zip(
                        wavelengths, hdul["FULL_DISK"].data["brightness"]):
                    plt.loglog(radius, data, label=wavelength)
                plt.ylabel(r"Surface Brightness ($\frac{erg}{cm^2rad^2\,s\,Hz}$)")
                plt.ylim([1e-10, None])
                legend = plt.legend()
                set_legend_color(legend, OPTIONS.plot.color.background)
                plt.xlabel(r"Radius (mas)")

            # TODO: Make this into the full disk
            elif data_type == "flux":
                stellar_flux = hdul["STAR"].data["flux"]
                inner_flux = hdul["INNER_RING"].data["total_flux"]
                outer_flux = hdul["OUTER_RING"].data["total_flux"]
                total_flux = stellar_flux+inner_flux+outer_flux
                plt.plot(wavelengths, total_flux, marker="o")
                plt.ylabel("Flux (Jy)")
                plt.xlabel(r"$\lambda$ ($\mathrm{\mu}$m)")

            elif data_type == "density":
                plt.loglog(radius, hdul["FULL_DISK"].data["surface_density"][0])
                plt.ylabel(r"Surface Density (g/cm$^2$)")
                plt.xlabel(r"Radius (mas)")

            elif data_type == "emissivity":
                for wavelength, data in zip(
                        wavelengths, hdul["FULL_DISK"].data["thickness"]):
                    plt.loglog(radius, data, label=wavelength)
                plt.ylabel(r"Emissivity (a.u.)")
                legend = plt.legend()
                set_legend_color(legend, OPTIONS.plot.color.background)
                plt.xlabel(r"Radius (mas)")

            elif data_type == "depth":
                nwl = len(wavelengths)
                optical_depth = hdul["FULL_DISK"].data["surface_density"]\
                        *(hdul["FULL_DISK"].data["kappa_abs"].reshape(nwl, 1)\
                          +hdul["FULL_DISK"].data["kappa_cont"].reshape(nwl, 1)*factor)
                for wavelength, data in zip(wavelengths, optical_depth):
                    plt.plot(radius, -np.log(1-data), label=wavelength)
                plt.ylabel(r"Optical depth (a.u.)")
                legend = plt.legend()
                set_legend_color(legend, OPTIONS.plot.color.background)
                plt.xscale("log")
                plt.xlabel(r"Radius (mas)")

    if title is not None:
        plt.title(title)

    if savefig is not None:
        plt.savefig(savefig, format=Path(savefig).suffix[1:],
                    dpi=OPTIONS.plot.dpi)
    plt.close()


def plot_all(model_file: Path,
             pixel_size: u.mas,
             wavelengths: u.um,
             zoom: Optional[float] = 25,
             save_dir: Optional[Path] = None) -> None:
    """Plots all the plots.

    Parameters
    ----------
    model_file : pathlib.Path
    pixel_size : astropy.units.Quantity
    wavelengths : astropy.units.Quantity
    zoom : float
    """
    for wavelength in wavelengths:
        wavelength = wavelength.value
        plot_model(model_file, data_type="image",
                   wavelength=wavelength,
                   pixel_size=pixel_size.value,
                   savefig=f"image{wavelength:.2f}.pdf", zoom=zoom)
    plot_model(model_file, data_type="temperature",
               savefig=save_dir / "temperature.pdf")
    plot_model(model_file, data_type="flux",
               savefig=save_dir / "flux.pdf")
    plot_model(model_file, data_type="brightness",
               savefig=save_dir / "brightness.pdf")
    plot_model(model_file, data_type="density",
               savefig=save_dir / "density.pdf")
    plot_model(model_file, data_type="emissivity",
               savefig=save_dir / "emissivity.pdf")
    plot_model(model_file, data_type="depth",
               factor=0.505, savefig=save_dir / "depth.pdf")


# TODO: Implement here Sebastian's code
# def add_subplot_axes(ax, rect, axisbg='w'):
#     axins = ax.inset_axes([0.5, 0.025, 0.49, 0.43])

#     yhat_basaltic = savitzky_golay(fpfs_basaltic, 25, 4) # window size 51, polynomial order 3
#     axins.plot(wvl_surf,yhat_basaltic, c='darkred', lw=2, label='basaltic')
#     yhat_ultramafic = savitzky_golay(fpfs_ultramafic, 25, 4) # window size 51, polynomial order 3
#     axins.plot(wvl_surf,yhat_ultramafic, c='r', ls=':', lw=2, label='ultramafic')
#     yhat_granitoid = savitzky_golay(fpfs_granitoid, 25, 4) # window size 51, polynomial order 3
#     axins.plot(wvl_surf,yhat_granitoid, c='orange', ls=':', lw=2, label='granitoid')
    # axins.errorbar([4.5], [380], xerr= [0.5], yerr = [40], fmt='.', c='k', label='Spitzer', 
    #              markeredgecolor='k', markerfacecolor='w', markersize=12)


def plot_datapoints(
        axarr, wavelengths: u.um,
        components: Optional[List] = None,
        data_to_plot: Optional[List[str]] = OPTIONS.fit.data,
        norm: Optional[np.ndarray] = None,
        wavelength_range: Optional[List[float]] = None,
        plot_nan: Optional[bool] = False,
        colormap: Optional[str] = OPTIONS.plot.color.colormap) -> None:
    """Plots the deviation of a model from real data of an object for
    total flux, visibilities and closure phases.

    Parameters
    ----------
    inclination : astropy.units.one
        The axis ratio.
    pos_angle : astropy.units.deg
        The position angle.
    data_to_plot : list of str, optional
        The data to plot. The default is OPTIONS.fit.data.
    savefig : pathlib.Path, optional
        The save path. The default is None.
    plot_nan : bool, optional
        If True plots the model values at points where the real data
        is nan (not found in the file for the wavelength specified).
    """
    colormap = get_colormap(colormap)
    hline_color = "gray" if OPTIONS.plot.color.background == "white"\
        else "white"

    flux, t3 = OPTIONS.data.flux, OPTIONS.data.t3
    vis = OPTIONS.data.vis if "vis" in OPTIONS.fit.data\
        else OPTIONS.data.vis2

    pos_angle, inclination = components[0].pa(), components[1].inc()
    flux_model, vis_model, t3_model = compute_observables(components)

    effective_baselines, _ = compute_effective_baselines(
        vis.ucoord, vis.vcoord, inclination, pos_angle)

    if "t3" in data_to_plot:
        longest_baselines, _ = compute_effective_baselines(
            t3.u123coord, t3.v123coord, inclination, pos_angle, longest=True)

    errorbar_params = OPTIONS.plot.errorbar
    scatter_params = OPTIONS.plot.scatter
    if OPTIONS.plot.color.background == "black":
        errorbar_params.markeredgecolor = "white"
        scatter_params.edgecolor = "white"

    for index, wavelength in enumerate(wavelengths):
        if wavelength_range is not None:
            if not (wavelength_range[0] <= wavelength <= wavelength_range[1]):
                continue

        effective_baselines_mlambda = (effective_baselines/wavelength.value)[1:]
        if "t3" in data_to_plot:
            longest_baselines_mlambda = (longest_baselines/wavelength.value)[1:]
        color = colormap(norm(wavelength.value))
        errorbar_params.color = scatter_params.color = color

        for key in data_to_plot:
            ax_key = "vis" if key in ["vis", "vis2"] else key
            upper_ax, lower_ax = axarr[ax_key]
            set_axes_color(upper_ax, OPTIONS.plot.color.background)
            set_axes_color(lower_ax, OPTIONS.plot.color.background)

            if key == "flux":
                if not plot_nan:
                    nan_flux = ~np.isnan(flux.value[index])
                else:
                    nan_flux = None

                upper_ax.errorbar(
                        wavelength.value.repeat(flux.err[index].size)[nan_flux],
                        flux.value[index][nan_flux],
                        flux.err[index][nan_flux],
                        fmt="o", **vars(errorbar_params))
                upper_ax.scatter(
                        wavelength.value.repeat(flux.err[index].size)[nan_flux],
                        flux_model[index][nan_flux],
                        marker="X", **vars(scatter_params))
                lower_ax.scatter(
                        wavelength.value.repeat(flux.err[index].size)[nan_flux],
                        flux.value[index][nan_flux]-flux_model[index][nan_flux],
                        marker="o", **vars(scatter_params))
                lower_ax.axhline(y=0, color=hline_color, linestyle='--')

            if key in ["vis", "vis2"]:
                if not plot_nan:
                    nan_vis = ~np.isnan(vis.value[index])
                else:
                    nan_vis = None

                upper_ax.errorbar(
                        effective_baselines_mlambda.value[nan_vis],
                        vis.value[index][nan_vis],
                        vis.err[index][nan_vis],
                        fmt="o", **vars(errorbar_params))
                upper_ax.scatter(
                        effective_baselines_mlambda.value[nan_vis],
                        vis_model[index][nan_vis],
                        marker="X", **vars(scatter_params))
                lower_ax.scatter(
                        effective_baselines_mlambda.value[nan_vis],
                        vis.value[index][nan_vis]-vis_model[index][nan_vis],
                        marker="o", **vars(scatter_params))
                lower_ax.axhline(y=0, color=hline_color, linestyle='--')

            if key == "t3":
                if not plot_nan:
                    nan_t3 = ~np.isnan(t3.value[index])
                else:
                    nan_t3 = None

                upper_ax.errorbar(
                        longest_baselines_mlambda.value[nan_t3],
                        t3.value[index][nan_t3], t3.err[index][nan_t3],
                        fmt="o", **vars(errorbar_params))
                upper_ax.scatter(
                        longest_baselines_mlambda.value[nan_t3],
                        t3_model[index][nan_t3],
                        marker="X", **vars(scatter_params))
                upper_ax.axhline(y=0, color=hline_color, linestyle='--')
                lower_ax.scatter(longest_baselines_mlambda.value[nan_t3],
                                 restrict_phase(
                                     t3.value[index][nan_t3]-t3_model[index][nan_t3]),
                                 marker="o", **vars(scatter_params))
                lower_ax.axhline(y=0, color=hline_color, linestyle='--')

    if "flux" in data_to_plot:
        axarr["flux"][0].set_xticks(wavelengths.value)
        axarr["flux"][1].set_xticks(wavelengths.value)
        axarr["flux"][1].set_xticklabels(wavelengths.value, rotation=45)
    errorbar_params.color = None


def plot_fit(components: Optional[List] = None,
             data_to_plot: Optional[List[str]] = None,
             cmap: Optional[str] = OPTIONS.plot.color.colormap,
             ylimits: Optional[Dict[str, List[float]]] = {},
             wavelength_range: Optional[List[float]] = None,
             plot_nan: Optional[bool] = False,
             title: Optional[str] = None,
             savefig: Optional[Path] = None):
    """Plots the deviation of a model from real data of an object for
    total flux, visibilities and closure phases.

    Parameters
    ----------
    inclination : astropy.units.one
        The axis ratio.
    pos_angle : astropy.units.deg
        The position angle.
    data_to_plot : list of str, optional
        The data to plot. The default is OPTIONS.fit.data.
    cmap : str, optional
        The colormap.
    plot_nan : bool, optional
        If True plots the model values at points where the real data
        is nan (not found in the file for the wavelength specified).
    title : str, optional
        The title. The default is None.
    savefig : pathlib.Path, optional
        The save path. The default is None.
    """
    data_to_plot = OPTIONS.fit.data\
        if data_to_plot is None else data_to_plot
    wavelengths = OPTIONS.fit.wavelengths
    norm = mcolors.Normalize(
            vmin=wavelengths[0].value, vmax=wavelengths[-1].value)

    data_types, nplots = [], 0
    for key in data_to_plot:
        if key in ["vis", "vis2"] and "vis" not in data_types:
            data_types.append("vis")
        else:
            data_types.append(key)
        nplots += 1

    figsize = (16, 5) if nplots == 3 else ((12, 5) if nplots == 2 else None)
    fig = plt.figure(
            figsize=figsize, facecolor=OPTIONS.plot.color.background)
    gs = GridSpec(2, nplots, height_ratios=[3, 1])
    axarr = {key: value for key, value in zip(
        data_types, [[fig.add_subplot(gs[j, i], facecolor=OPTIONS.plot.color.background)
                      for j in range(2)] for i in range(nplots)])}

    plot_datapoints(axarr, wavelengths, components=components,
                    wavelength_range=wavelength_range,
                    norm=norm, data_to_plot=data_to_plot,
                    plot_nan=plot_nan, colormap=cmap)

    sm = cm.ScalarMappable(cmap=get_colormap(cmap), norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axarr[data_types[-1]])
    cbar.set_ticks(wavelengths.value)
    cbar.set_ticklabels([f"{wavelength:.1f}"
                         for wavelength in wavelengths.value])
    if OPTIONS.plot.color.background == "black":
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")
        for spine in cbar.ax.spines.values():
            spine.set_edgecolor("white")
    opposite_color = "white" if OPTIONS.plot.color.background == "black"\
        else "black"
    cbar.set_label(label=r"$\lambda$ ($\mathrm{\mu}$m)",
                   color=opposite_color)

    label_color = "lightgray" if OPTIONS.plot.color.background == "black"\
        else "k"
    dot_label = mlines.Line2D([], [], color=label_color, marker="o",
                              linestyle="None", label="Data", alpha=0.6)
    x_label = mlines.Line2D([], [], color=label_color, marker="X",
                            linestyle="None", label="Model")
    tick_settings = {"axis": "x", "which": "both",
                     "bottom": True, "top": False,
                     "labelbottom": False}

    for key in data_to_plot:
        ax_key = "vis" if key in ["vis", "vis2"] else key
        upper_ax, lower_ax = axarr[ax_key]

        if key == "flux":
            lower_ax.set_xlabel(r"$\lambda$ ($\mathrm{\mu}$m)")
            lower_ax.set_ylabel("Residuals (Jy)")
            upper_ax.tick_params(**tick_settings)
            upper_ax.set_ylabel("Fluxes (Jy)")
            if "flux" in ylimits:
                upper_ax.set_ylim(ylimits["flux"])
            else:
                upper_ax.set_ylim([0, None])
            if not len(axarr) > 1:
                legend = upper_ax.legend(handles=[dot_label, x_label])
                set_legend_color(legend, OPTIONS.plot.color.background)

        if key in ["vis", "vis2"]:
            lower_ax.set_xlabel(r"$\mathrm{B}_{\mathrm{eff}}$ (M$\lambda$)")

            if key == "vis":
                if OPTIONS.model.output != "normed":
                    y_label = "Correlated fluxes (Jy)"
                    upper_ax.set_ylim([0, None])
                    unit = "Jy"
                else:
                    y_label = "Visibilities (Normalized)"
                    unit = "Normalized"
                    upper_ax.set_ylim([0, 1])

                residual_label = f"Residuals ({unit})"
                if "vis" in ylimits:
                    upper_ax.set_ylim(ylimits["vis"])
            else:
                residual_label = "Residuals (Normalized)"
                y_label = "Visibilities Squared (Normalized)"
                if "vis2" in ylimits:
                    upper_ax.set_ylim(ylimits["vis2"])
                else:
                    upper_ax.set_ylim([0, 1])

            upper_ax.set_xlim([0, None])
            lower_ax.set_xlim([0, None])
            lower_ax.set_ylabel(residual_label)
            upper_ax.set_ylabel(y_label)
            upper_ax.tick_params(**tick_settings)
            upper_ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

            if not len(axarr) > 1:
                legend = upper_ax.legend(handles=[dot_label, x_label])
                set_legend_color(legend, OPTIONS.plot.color.background)

        if key == "t3":
            upper_ax.set_ylabel(r"Closure Phases ($^\circ$)")
            lower_ax.set_xlabel(r"$\mathrm{B}_{\mathrm{max}}$ (M$\lambda$)")
            lower_ax.set_ylabel(r"Residuals ($^\circ$)")
            t3 = OPTIONS.data.t3
            nan_t3 = np.isnan(t3.value)
            lower_bound = t3.value[~nan_t3].min()
            lower_bound += lower_bound*0.25
            upper_bound = t3.value[~nan_t3].max()
            upper_bound += upper_bound*0.25
            upper_ax.tick_params(**tick_settings)
            if "t3" in ylimits:
                upper_ax.set_ylim(ylimits["t3"])
            else:
                upper_ax.set_ylim([lower_bound, upper_bound])

            upper_ax.set_xlim([0, None])
            lower_ax.set_xlim([0, None])
            legend = upper_ax.legend(handles=[dot_label, x_label])
            set_legend_color(legend, OPTIONS.plot.color.background)

    if title is not None:
        plt.title(title)

    if savefig is not None:
        plt.savefig(savefig, format=Path(savefig).suffix[1:],
                    dpi=OPTIONS.plot.dpi)
    else:
        plt.show()
    plt.close()


def plot_overview(data_to_plot: Optional[List[str]] = None,
                  colormap: Optional[str] = OPTIONS.plot.color.colormap,
                  wavelength_range: Optional[List[float]] = None,
                  ylimits: Optional[Dict[str, List[float]]] = {},
                  title: Optional[str] = None,
                  raxis: Optional[bool] = False,
                  inclination: Optional[float] = None,
                  pos_angle: Optional[float] = None,
                  savefig: Optional[Path] = None) -> None:
    """Plots an overview over the total data for baselines [Mlambda].

    Parameters
    ----------
    data_to_plot : list of str, optional
        The data to plot. The default is OPTIONS.fit.data.
    savefig : pathlib.Path, optional
        The save path. The default is None.
    """
    data_to_plot = OPTIONS.fit.data\
        if data_to_plot is None else data_to_plot
    wavelengths = OPTIONS.fit.wavelengths
    norm = mcolors.Normalize(
            vmin=wavelengths[0].value, vmax=wavelengths[-1].value)

    data_types, nplots = [], 0
    for key in data_to_plot:
        if key in ["vis", "vis2"] and "vis" not in data_types:
            data_types.append("vis")
        else:
            data_types.append(key)
        nplots += 1

    figsize = (15, 5) if nplots == 3 else ((12, 5) if nplots == 2 else None)
    fig, axarr = plt.subplots(1, nplots, figsize=figsize,
                              tight_layout=True,
                              facecolor=OPTIONS.plot.color.background)
    axarr = axarr.flatten() if isinstance(axarr, np.ndarray) else [axarr]
    axarr = dict(zip(data_types, axarr))

    colormap = get_colormap(colormap)
    hline_color = "gray" if OPTIONS.plot.color.background == "white"\
        else "white"

    flux, t3 = OPTIONS.data.flux, OPTIONS.data.t3
    vis = OPTIONS.data.vis if "vis" in OPTIONS.fit.data\
        else OPTIONS.data.vis2

    # TODO: Set the color somewhere centrally so all plots are the same color.
    errorbar_params = OPTIONS.plot.errorbar
    if OPTIONS.plot.color.background == "black":
        errorbar_params.markeredgecolor = "white"

    for index, wavelength in enumerate(wavelengths):
        if wavelength_range is not None:
            if not (wavelength_range[0] <= wavelength <= wavelength_range[1]):
                continue

        effective_baselines, _ = compute_effective_baselines(
            vis.ucoord, vis.vcoord, inclination, pos_angle)
        effective_baselines_mlambda = (effective_baselines/wavelength.value)[1:]

        if "t3" in data_to_plot:
            longest_baselines, _ = compute_effective_baselines(
                t3.u123coord, t3.v123coord, inclination, pos_angle, longest=True)
            longest_baselines, _ = compute_effective_baselines(
                t3.u123coord, t3.v123coord, longest=True)
            longest_baselines_mlambda = (longest_baselines / wavelength.value)[1:]

        color = colormap(norm(wavelength.value))
        errorbar_params.color = color

        for key in data_to_plot:
            ax_key = "vis" if key in ["vis", "vis2"] else key
            ax = axarr[ax_key]
            if key == "flux":
                ax.errorbar(
                    wavelength.value.repeat(flux.err[index].size),
                    flux.value[index], flux.err[index],
                    fmt="o", **vars(errorbar_params))

            if key in ["vis", "vis2"]:
                ax.errorbar(
                    effective_baselines_mlambda.value,
                    vis.value[index], vis.err[index],
                    fmt="o", **vars(errorbar_params))

            if key == "t3":
                ax.errorbar(
                    longest_baselines_mlambda.value,
                    t3.value[index], t3.err[index],
                    fmt="o", **vars(errorbar_params))
                ax.axhline(y=0, color=hline_color, linestyle='--')

    if "flux" in data_to_plot:
        axarr["flux"].set_xticks(wavelengths.value)
        axarr["flux"].set_xticklabels(wavelengths.value, rotation=45)

    errorbar_params.color = None

    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axarr[data_types[-1]])
    cbar.set_ticks(wavelengths.value)
    cbar.set_ticklabels([f"{wavelength:.1f}"
                         for wavelength in wavelengths.value])
    if OPTIONS.plot.color.background == "black":
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")
        for spine in cbar.ax.spines.values():
            spine.set_edgecolor("white")
    opposite_color = "white" if OPTIONS.plot.color.background == "black"\
        else "black"
    cbar.set_label(label=r"$\lambda$ ($\mathrm{\mu}$m)",
                   color=opposite_color)

    for key in data_to_plot:
        ax_key = "vis" if key in ["vis", "vis2"] else key
        ax = axarr[ax_key]
        set_axes_color(ax, OPTIONS.plot.color.background)

        if key == "flux":
            ax.set_xlabel(r"$\lambda$ ($\mathrm{\mu}$m)")
            ax.set_ylabel("Fluxes (Jy)")
            if "flux" in ylimits:
                ax.set_ylim(ylimits["flux"])
            else:
                ax.set_ylim([0, None])

        if inclination is not None:
            label = r"$\mathrm{B}_\mathrm{eff}$ (M$\lambda$)"
        else:
            label = r"$\mathrm{B}$ (M$\lambda$)"

        if key == "vis":
            ax.set_xlabel(label)
            if OPTIONS.model.output != "normed":
                label = "Correlated fluxes (Jy)"
            else:
                label = "Visibilities (Normalized)"

            ax.set_ylabel(label)
            if "vis" in ylimits:
                ax.set_ylim(ylimits["vis"])
            ax.set_ylim([0, None])
            ax.set_xlim([0, None])
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

        if key == "vis2":
            ax.set_xlabel(label)
            ax.set_ylabel("Squared Visibilities (Normalized)")
            if "vis2" in ylimits:
                ax.set_ylim(ylimits["vis2"])
            else:
                ax.set_ylim([0, 1])
            ax.set_xlim([0, None])
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

        if key == "t3":
            ax.set_xlabel(r"$\mathrm{B}_{\mathrm{max}}$ (M$\lambda$)")
            ax.set_ylabel(r"Closure Phases ($^\circ$)")
            nan_t3 = np.isnan(t3.value)
            lower_bound = t3.value[~nan_t3].min()
            lower_bound += lower_bound*0.25
            upper_bound = t3.value[~nan_t3].max()
            upper_bound += upper_bound*0.25
            if "t3" in ylimits:
                ax.set_ylim(ylimits["t3"])
            else:
                ax.set_ylim([lower_bound, upper_bound])
            ax.set_xlim([0, None])

    if title is not None:
        plt.title(title)

    if raxis:
        return fig, axarr

    if savefig is not None:
        plt.savefig(savefig, format=Path(savefig).suffix[1:],
                    dpi=OPTIONS.plot.dpi)
    else:
        plt.show()
    plt.close()


def plot_target(target: str,
                wavelength_range: Optional[List[float]] = None,
                radius: Optional[float] = 1,
                ax: Optional[plt.Axes] = None,
                title: Optional[str] = None,
                filters: Optional[List[str]] = None,
                show_legend: Optional[bool] = False,
                savefig: Optional[Path] = None) -> None:
    """Plots the target's photometry from Vizier.

    Parameters
    ----------
    target : str
    radius : float, optional
    """
    sed = Table.read(f"https://vizier.cds.unistra.fr/viz-bin/sed?-c={target}&-c.rs={radius}")

    if ax is None:
        _ = plt.figure(facecolor=OPTIONS.plot.color.background,
                       tight_layout=True)
        ax = plt.axes(facecolor=OPTIONS.plot.color.background)
        set_axes_color(ax, OPTIONS.plot.color.background)
        ax.set_xlabel(r"$\lambda$ ($\mathrm{\mu}$m)")
        ax.set_ylabel("Flux (Jy)")
        ax.set_title(title)
    else:
        fig = None

    colors = OPTIONS.plot.color.list
    if filters is not None:
        for filter in filters:
            filtered_sed = sed[[filter_name.startswith(filter)
                                for filter_name in sed['sed_filter']]]

            for index, tabname in enumerate(set(filtered_sed['_tabname'])):
                subset = filtered_sed[filtered_sed['_tabname'] == tabname]
                frequency, flux = subset["sed_freq"], subset["sed_flux"]
                wavelength = (const.c/(frequency).to(u.Hz)).to(u.um)

                if wavelength_range is not None:
                    indices = np.where((wavelength > wavelength_range[0]) &
                                       (wavelength < wavelength_range[1]))[0]

                    if indices.size == 0:
                        continue

                    wavelength = wavelength[indices]
                    flux = flux[indices]

                ax.scatter(wavelength, flux, label=f"{filter}, {tabname}",
                           color=colors[index])
    else:
        for index, tabname in enumerate(set(sed['_tabname'])):
            subset = sed[sed['_tabname'] == tabname]
            frequency, flux = subset["sed_freq"], subset["sed_flux"]
            wavelength = (const.c/(frequency).to(u.Hz)).to(u.um)

            if wavelength_range is not None:
                indices = np.where((wavelength > wavelength_range[0]) &
                                   (wavelength < wavelength_range[1]))[0]

                if indices.size == 0:
                    continue

                wavelength = wavelength[indices]
                flux = flux[indices]

            ax.scatter(wavelength, flux, color=colors[index])

    if show_legend:
        ax.legend()

    if savefig is not None:
        plt.savefig(savefig, format=Path(savefig).suffix[1:],
                    dpi=OPTIONS.plot.dpi)

    if fig is not None:
        plt.close()


def plot_observables(wavelength_range: u.um,
                     components: List[Component],
                     component_labels: List[str],
                     save_dir: Optional[Path] = None) -> None:
    """Plots the observables of the model.

    Parameters
    ----------
    wavelength_range : astropy.units.m
    """
    save_dir = Path.cwd() if save_dir is None else save_dir
    wavelength = np.linspace(
        wavelength_range[0], wavelength_range[1], OPTIONS.plot.dim)
    flux, vis, t3, vis_comps = compute_observables(
        components, wavelength=wavelength, rcomponents=True)

    if "flux" in OPTIONS.fit.data:
        _ = plt.figure(facecolor=OPTIONS.plot.color.background,
                       tight_layout=True)
        ax = plt.axes(facecolor=OPTIONS.plot.color.background)
        set_axes_color(ax, OPTIONS.plot.color.background)
        names = [re.findall(r"(\d{4}-\d{2}-\d{2})", readout.fits_file.name)[0] for readout in OPTIONS.data.readouts]
        sorted_readouts = np.array(OPTIONS.data.readouts.copy())[np.argsort(names)].tolist()

        values = [flux[:, 0]]
        for name, readout in zip(np.sort(names), sorted_readouts):
            if readout.flux.value.size == 0:
                continue

            readout_wavelength = readout.wavelength.value
            readout_flux, readout_err = readout.flux.value.flatten(), readout.flux.err.flatten()
            if "HAW" in readout.fits_file.name:
                indices_high = np.where((readout_wavelength >= 4.55) & (readout_wavelength <= 4.9))
                indices_low = np.where((readout_wavelength >= 3.1) & (readout_wavelength <= 3.9))
                indices = np.hstack((indices_high, indices_low))
                readout_wavelength = readout_wavelength[indices].flatten()
                readout_flux, readout_err = readout_flux[indices].flatten(), readout_err[indices].flatten()

            lower_err, upper_err = readout_flux - readout_err, readout_flux + readout_err
            line = ax.plot(readout_wavelength, readout_flux, label=name)
            ax.fill_between(readout_wavelength, lower_err, upper_err, color=line[0].get_color(), alpha=0.5)
            values.append(readout_flux)

        ax.plot(wavelength, flux[:, 0], label="Model", color="black")

        max_value = np.concatenate(values).max()
        ax.set_xlabel(r"$\lambda$ ($\mathrm{\mu}$m)")
        ax.set_ylabel("Flux (Jy)")
        ax.set_ylim([0, max_value + 0.1*max_value])
        handles, labels = ax.get_legend_handles_labels()
        custom_handles = [handles[0], handles[1], Line2D([0], [0], color="white", label="..."), handles[-1]]
        custom_labels = [labels[0], labels[1], "...", labels[-1]]
        ax.legend(custom_handles, custom_labels)
        plt.savefig(save_dir / "sed.pdf", format="pdf")
        plt.close()

    vis_data = OPTIONS.data.vis if "vis" in OPTIONS.fit.data\
        else OPTIONS.data.vis2

    effective_baselines, baseline_angles = compute_effective_baselines(
            vis_data.ucoord, vis_data.vcoord,
            components[1].inc(), components[1].pa(), rzero=False)

    num_plots = len(effective_baselines)
    cols = int(str(num_plots)[:int(np.floor(np.log10(num_plots)))])
    rows = int(np.ceil(num_plots/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(30, 30),
                             facecolor=OPTIONS.plot.color.background,
                             sharex=True, constrained_layout=True)
    axes = axes.flatten()
    if "vis" in OPTIONS.fit.data:
        if OPTIONS.model.output != "normed":
            y_label = "Correlated fluxes (Jy)"
            ylims = [None, None]
        else:
            y_label = "Visibilities (Normalized)"
            ylims = [0, 1]
    else:
        y_label = "Visibilities Squared (Normalized)"
        ylims = [0, 1]

    for index, (baseline, baseline_angle) in enumerate(
            zip(effective_baselines, baseline_angles)):
        ax = axes[index]
        set_axes_color(ax, OPTIONS.plot.color.background)
        ax.plot(wavelength, vis[:, index],
                label=rf"B={baseline.value:.2f} m, $\phi$={baseline_angle.value:.2f}$^\circ$")

        for comp_index, vis_comp in enumerate(vis_comps):
            ax.plot(wavelength, vis_comp[:, index], label=component_labels[comp_index])

        ax.set_ylim(ylims)
        ax.legend()

    fig.subplots_adjust(left=0.2, bottom=0.2)
    fig.text(0.5, 0.04, r"$\lambda$ ($\mathrm{\mu}$m)", ha="center", fontsize=16)
    fig.text(0.04, 0.5, y_label, va="center", rotation="vertical", fontsize=16)
    plt.savefig(save_dir / "vis_vs_baseline.pdf", format="pdf")
    plt.close()

    if "t3" in OPTIONS.fit.data:
        effective_baselines, baseline_angles = compute_effective_baselines(
                OPTIONS.data.t3.u123coord, OPTIONS.data.t3.v123coord,
                components[1].inc(), components[1].pa(), longest=True, rzero=False)

        num_plots = len(effective_baselines)
        cols = int(str(num_plots)[:int(np.floor(np.log10(num_plots)))])
        rows = int(np.ceil(num_plots/cols))
        fig, axes = plt.subplots(rows, cols, figsize=(30, 30),
                                 facecolor=OPTIONS.plot.color.background,
                                 sharex=True, constrained_layout=True)
        axes = axes.flatten()
        for index, (baseline, baseline_angle) in enumerate(
            zip(effective_baselines, baseline_angles)):
            ax = axes[index]
            set_axes_color(ax, OPTIONS.plot.color.background)
            ax.plot(wavelength, t3[:, index], label=f"B={baseline.value:.2f} m")
            ax.legend()

        fig.subplots_adjust(left=0.2, bottom=0.2)
        fig.text(0.5, 0.04, r"$\lambda$ ($\mathrm{\mu}$m)", ha="center", fontsize=16)
        fig.text(0.04, 0.5, y_label, va="center", rotation="vertical", fontsize=16)
        plt.savefig(save_dir / "t3_vs_baseline.pdf", format="pdf")
        plt.close()


# TODO: Finish this function
def plot_sth(dim: int, wavelength: Optional[u.Quantity[u.um]] = None):
    wavelength = u.Quantity(wavelength, u.um) if wavelength is not None\
        else OPTIONS.fit.wavelengths

    for index, component in enumerate(components):
        if component.shortname not in ["Star", "Point"]:
            component.dim.value = dim
            radius = np.tile(component.compute_internal_grid(), (wavelength.size, 1))

            temperature = component.compute_temperature(radius)
            surface_density = component.compute_surface_density(radius)
            optical_depth = component.compute_optical_depth(
                    radius, wavelength[:, np.newaxis])
            emissivity = component.compute_emissivity(
                    radius, wavelength[:, np.newaxis])
            brightness = component.compute_intensity(
                    radius, wavelength[:, np.newaxis])

        flux = component.compute_flux(wavelength[:, np.newaxis])
