from datetime import datetime
from typing import Optional, Any, Dict, List
from pathlib import Path

import astropy.units as u
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
# from dynesty import plotting as dyplot
from matplotlib.gridspec import GridSpec

from .component import Component
from .data import ReadoutFits
from .fitting import calculate_observables
from .model import Model
from .options import OPTIONS, get_colormap
from .utils import calculate_effective_baselines, restrict_phase

matplotlib.use('Agg')

HEADER_DICT = {
        "BUNIT": ("Jy", "Unit of original pixel value"),
        "BTYPE": ("Brightness", "Type of original pixel value"),
        "EXTEND": (True, "EXTEND"),
        "COMMENT": "Best fit model image per wavelength",
        }

def set_axes_color(ax: matplotlib.axes.Axes,
                   background_color: str) -> None:
    """Sets all the axes to the opposite color."""
    opposite_color = "white" if background_color == "black" else "black"
    ax.set_facecolor(background_color)
    ax.spines['bottom'].set_color(opposite_color)
    ax.spines['top'].set_color(opposite_color)
    ax.spines['right'].set_color(opposite_color)
    ax.spines['left'].set_color(opposite_color)
    ax.xaxis.label.set_color(opposite_color)
    ax.yaxis.label.set_color(opposite_color)
    ax.tick_params(axis='x', colors=opposite_color)
    ax.tick_params(axis='y', colors=opposite_color)


def set_legend_color(legend: matplotlib.legend.Legend,
                     background_color: str) -> None:
    """Sets the legend color."""
    opposite_color = "white" if background_color == "black" else "black"
    plt.setp(legend.get_texts(), color=opposite_color)
    legend.get_frame().set_facecolor(background_color)


def plot_corner(sampler: np.ndarray, labels: List[str],
                units: Optional[List[str]] = None,
                discard: Optional[int] = 0,
                savefig: Optional[Path] = None) -> None:
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
    if OPTIONS["fit.method"] == "emcee":
        samples = sampler.get_chain(discard=discard, flat=True)
        corner.corner(samples, show_titles=True,
                      labels=labels, plot_datapoints=True,
                      quantiles=[x/100 for x in OPTIONS["fit.quantiles"]],
                      title_kwargs={"fontsize": 12})
    else:
        dyplot.cornerplot(sampler.results)

    if savefig is not None:
        plt.savefig(savefig, format="pdf")
    else:
        plt.show()
    plt.close()


def plot_chains(sampler: np.ndarray, labels: List[str],
                units: Optional[List[str]] = None,
                discard: Optional[int] = 0,
                savefig: Optional[Path] = None) -> None:
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


def save_fits(dim: int, pixel_size: u.mas, distance: u.pc,
              wavelengths: List[u.Quantity[u.um]],
              components: List[Component],
              component_labels: List[str],
              opacities: List[np.ndarray] = None,
              savefits: Optional[Path] = None,
              options: Optional[Dict[str, Any]] = None,
              object_name: Optional[str] = None,
              nwalkers: Optional[int] = None,
              nsteps: Optional[int] = None,
              ncores: Optional[int] = None) -> None:
    """Saves a (.fits)-file of the model with all the information on the
    parameter space."""
    pixel_size = pixel_size if isinstance(pixel_size, u.Quantity)\
        else pixel_size*u.mas
    wavelengths = wavelengths if isinstance(wavelengths, u.Quantity)\
        else wavelengths*u.um
    distance = distance if isinstance(distance, u.Quantity) else distance*u.pc

    model, images = Model(components), []
    for wavelength in wavelengths:
        images.append(model.calculate_image(dim, pixel_size, wavelength))
    images = np.array(images)

    tables = []
    for index, component in enumerate(components):
        table_header = fits.Header()

        table_header["COMP"] = component.name
        if options is not None:
            if "model.gridtype" in options:
                table_header["GRIDTYPE"] = (options["model.gridtype"], "The type of the grid")

        data = {"wavelength": wavelengths}
        if component.name != "Star":
            innermost_radius = component.params["rin0"]()\
                    if component.params["rin0"]() != 0 else component.params["rin"]()
            radius = component._calculate_internal_grid(dim)

            data["radius"] = [radius]*len(wavelengths)
            data["temperature"] = [component._temperature_profile_function(
                radius, innermost_radius)]*len(wavelengths)
            data["surface_density"] = [component._surface_density_profile_function(
                radius, innermost_radius)]*len(wavelengths)

            for wavelength in wavelengths:
                if "flux" not in data:
                    data["flux"] = []
                if "thickness" not in data:
                    data["thickness"] = []
                if "brightness" not in data:
                    data["brightness"] = []
                data["flux"].append(component.calculate_flux(wavelength))
                data["thickness"].append(component._thickness_profile_function(
                        radius, innermost_radius, wavelength))
                data["brightness"].append(component._brightness_profile_function(
                        radius, wavelength))

        for wavelength in wavelengths:
            for parameter in component.params.values():
                if parameter.wavelength is None:
                    name = parameter.shortname.upper()
                    if name not in table_header:
                        description = f"[{parameter.unit}] {parameter.description}"
                        table_header[name] = (parameter().value, description)
                else:
                    if parameter.name not in data:
                        data[parameter.name] = [parameter(wavelength).value]
                    else:
                        data[parameter.name].append(parameter(wavelength).value)

        table = fits.BinTableHDU(
                Table(data=data),
                name="_".join(component_labels[index].split(" ")).upper(),
                header=table_header)
        tables.append(table)

    data = None
    for table in tables:
        if table.header["COMP"] == "Star":
            continue
        if data is None:
            data = {col.name: table.data[col.name] for col in table.columns}
            continue
        for column in table.columns:
            # TODO: Make calculation work for total flux
            if column.name in ["wavelength", "kappa_abs", "kappa_cont", "flux"]:
                continue
            if column.name == "radius":
                filler = np.tile(
                        np.linspace(data[column.name][0][-1], table.data[column.name][0][0], dim),
                        (table.data[column.name].shape[0], 1))
            else:
                filler = np.zeros(data[column.name].shape)
            data[column.name] = np.hstack((data[column.name],
                                           filler, table.data[column.name]))
    table = fits.BinTableHDU(Table(data=data), name="FULL_DISK")
    tables.append(table)

    if opacities is not None:
        data = {"wavelength": opacities[0].wavelength}
        for opacity in opacities:
            data[opacity.shortname] = opacity()
        tables.append(fits.BinTableHDU(Table(data=data), name="OPACITIES"))

    wcs = WCS(naxis=3)
    wcs.wcs.crpix = (*np.array(images.shape[:2]) // 2, len(wavelengths))
    wcs.wcs.cdelt = ([pixel_size.value, pixel_size.value, -1.0])
    wcs.wcs.crval = (0.0, 0.0, 1.0)
    wcs.wcs.ctype = ("RA---AIR", "DEC--AIR", "WAVELENGTHS")
    wcs.wcs.cunit = ("mas", "mas", "um")
    wcs.wcs.pc = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    header = wcs.to_header()

    header["NSTEP"] = (nsteps, "Number of steps for the fitting")
    header["NWALK"] = (nwalkers, "Numbers of walkers for the fitting")
    header["NCORE"] = (ncores, "Numbers of cores for the fitting")
    header["OBJECT"] = (object_name, "Name of the object")
    header["DATE"] = (f"{datetime.now()}", "Creation date")
    # header["LTM1_1"] = np.around(pixel_size_au.value, 5), "Pixel size for x-coordinate (au)"
    # header["LTM2_2"] = np.around(pixel_size_au.value, 5), "Pixel size for y-coordinate (au)"

    if options is not None:
        if "model.gridtype" in options:
            header["GRIDTYPE"] = (options["model.gridtype"], "The type of the grid")
        if "model.flux.factor" in options:
            header["FLXFACT"] = (options["model.flux.factor"],
                                 "The factor with which the flux is multiplied")

    hdu = fits.HDUList([fits.PrimaryHDU(images, header=header), *tables])
    hdu.writeto(savefits, overwrite=True)


# TODO: Make this plot multiple models for each wavelength
def plot_model(fits_file: Path, data_type: Optional[str] = "image",
               wavelength: Optional[float] = None,
               pixel_size: Optional[float] = None,
               factor: Optional[float] = None,
               zoom: Optional[int] = 30,
               colormap: Optional[str] = OPTIONS["plot.color.colormap"],
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
    _ = plt.figure(facecolor=OPTIONS["plot.color.background"])
    ax = plt.axes(facecolor=OPTIONS["plot.color.background"])
    set_axes_color(ax, OPTIONS["plot.color.background"])
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
                    plt.plot(radius, data, label=wavelength)
                plt.ylabel(r"Surface Brightness ($\frac{erg}{cm^2rad^2\,s\,Hz}$)")
                plt.yscale("log")
                plt.xscale("log")
                plt.ylim([1e-10, None])
                legend = plt.legend()
                set_legend_color(legend, OPTIONS["plot.color.background"])
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
                plt.plot(radius, hdul["FULL_DISK"].data["surface_density"][0])
                plt.ylabel(r"Surface Density (g/cm$^2$)")
                plt.yscale("log")
                plt.xscale("log")
                plt.xlabel(r"Radius (mas)")
            elif data_type == "thickness":
                for wavelength, data in zip(
                        wavelengths, hdul["FULL_DISK"].data["thickness"]):
                    plt.plot(radius, data, label=wavelength)
                plt.ylabel(r"Thickness (a.u.)")
                plt.yscale("log")
                plt.xscale("log")
                legend = plt.legend()
                set_legend_color(legend, OPTIONS["plot.color.background"])
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
                set_legend_color(legend, OPTIONS["plot.color.background"])
                plt.xscale("log")
                plt.xlabel(r"Radius (mas)")

    if title is not None:
        plt.title(title)

    if savefig is not None:
        plt.savefig(savefig, format=Path(savefig).suffix[1:],
                    dpi=OPTIONS["plot.dpi"])
    else:
        plt.show()
    plt.close()


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
        axarr, axis_ratio: u.one,
        pos_angle: u.deg, wavelengths: u.um,
        components: Optional[List] = None,
        data_to_plot: Optional[List[str]] = OPTIONS["fit.data"],
        norm: Optional[np.ndarray] = None,
        wavelength_range: Optional[List[float]] = None,
        colormap: Optional[str] = OPTIONS["plot.color.colormap"]) -> None:
    """Plots the deviation of a model from real data of an object for
    total flux, visibilities and closure phases.

    Parameters
    ----------
    axis_ratio : astropy.units.one
        The axis ratio.
    pos_angle : astropy.units.deg
        The position angle.
    data_to_plot : list of str, optional
        The data to plot. The default is OPTIONS["fit.data"].
    savefig : pathlib.Path, optional
        The save path. The default is None.
    """
    colormap = get_colormap(colormap)
    hline_color = "gray" if OPTIONS["plot.color.background"] == "white"\
        else "white"

    fluxes, fluxes_err =\
        OPTIONS["data.flux"], OPTIONS["data.flux_err"]

    if "vis" in OPTIONS["fit.data"]:
        vis, vis_err =\
            OPTIONS["data.corr_flux"], OPTIONS["data.corr_flux_err"]
        ucoord, vcoord =\
            OPTIONS["data.corr_flux.ucoord"], OPTIONS["data.corr_flux.vcoord"]
    else:
        vis, vis_err =\
            OPTIONS["data.vis"], OPTIONS["data.vis_err"]
        ucoord, vcoord =\
            OPTIONS["data.vis.ucoord"], OPTIONS["data.vis.vcoord"]

    cphases, cphases_err =\
        OPTIONS["data.cphase"], OPTIONS["data.cphase_err"]
    u123coord = OPTIONS["data.cphase.u123coord"]
    v123coord = OPTIONS["data.cphase.v123coord"]
    errorbar_params = OPTIONS["plot.errorbar"]
    scatter_params = OPTIONS["plot.scatter"]
    if OPTIONS["plot.color.background"] == "black":
        errorbar_params["markeredgecolor"] = "white"
        scatter_params["edgecolor"] = "white"

    for index, wavelength in enumerate(wavelengths):
        if wavelength_range is not None:
            if not (wavelength_range[0] <= wavelength <= wavelength_range[1]):
                continue
        effective_baselines = calculate_effective_baselines(
            ucoord[index], vcoord[index],
            axis_ratio, pos_angle)[0]
        longest_baselines = calculate_effective_baselines(
            u123coord[index], v123coord[index],
            axis_ratio, pos_angle)[0].max(axis=0)
        flux_model, corr_flux_model, cphase_model = calculate_observables(
                components, wavelength, ucoord[index],
                vcoord[index], u123coord[index], v123coord[index])

        vis_model = corr_flux_model if "vis" in data_to_plot\
            else corr_flux_model/flux_model
        effective_baselines_mlambda = effective_baselines/wavelength.value
        longest_baselines_mlambda = longest_baselines/wavelength.value
        color = colormap(norm(wavelength.value))
        errorbar_params["color"] = scatter_params["color"] = color

        for key in data_to_plot:
            ax_key = "vis" if key in ["vis", "vis2"] else key
            upper_ax, lower_ax = axarr[ax_key]
            set_axes_color(upper_ax, OPTIONS["plot.color.background"])
            set_axes_color(lower_ax, OPTIONS["plot.color.background"])

            if key == "flux":
                for flux, flux_err in zip(
                        fluxes[index], fluxes_err[index]):
                    upper_ax.errorbar(
                            wavelength.value, flux, flux_err,
                            fmt="o", **errorbar_params)
                    upper_ax.scatter(
                            wavelength.value, flux_model,
                            marker="X", **scatter_params)
                    lower_ax.scatter(
                            wavelength.value, flux-flux_model,
                            marker="o", **scatter_params)
                lower_ax.axhline(y=0, color=hline_color, linestyle='--')

            if key in ["vis", "vis2"]:
                upper_ax.errorbar(
                        effective_baselines_mlambda.value,
                        vis[index], vis_err[index],
                        fmt="o", **errorbar_params)
                upper_ax.scatter(
                        effective_baselines_mlambda.value, vis_model,
                        marker="X", **scatter_params)
                lower_ax.scatter(
                        effective_baselines_mlambda.value,
                        vis[index]-vis_model, marker="o", **scatter_params)
                lower_ax.axhline(y=0, color=hline_color, linestyle='--')

            if key == "t3phi":
                upper_ax.errorbar(
                        longest_baselines_mlambda.value,
                        cphases[index], cphases_err[index],
                        fmt="o", **errorbar_params)
                upper_ax.scatter(
                        longest_baselines_mlambda.value, cphase_model,
                        marker="X", **scatter_params)
                upper_ax.axhline(y=0, color=hline_color, linestyle='--')
                lower_ax.scatter(longest_baselines_mlambda.value,
                                 restrict_phase(cphases[index]-cphase_model),
                                 marker="o", **scatter_params)
                lower_ax.axhline(y=0, color=hline_color, linestyle='--')

    if "flux" in data_to_plot:
        axarr["flux"][0].set_xticks(wavelengths.value)
        axarr["flux"][1].set_xticks(wavelengths.value)
        axarr["flux"][1].set_xticklabels(wavelengths.value, rotation=45)
    errorbar_params["color"] = ""


def plot_fit(axis_ratio: u.one, pos_angle: u.deg,
             components: Optional[List] = None,
             data_to_plot: Optional[List[str]] = None,
             colormap: Optional[str] = OPTIONS["plot.color.colormap"],
             ylimits: Optional[Dict[str, List[float]]] = {},
             wavelength_range: Optional[List[float]] = None,
             title: Optional[str] = None,
             savefig: Optional[Path] = None):
    """Plots the deviation of a model from real data of an object for
    total flux, visibilities and closure phases.

    Parameters
    ----------
    axis_ratio : astropy.units.one
        The axis ratio.
    pos_angle : astropy.units.deg
        The position angle.
    data_to_plot : list of str, optional
        The data to plot. The default is OPTIONS["fit.data"].
    colormap : str, optional
        The colormap.
    title : str, optional
        The title. The default is None.
    savefig : pathlib.Path, optional
        The save path. The default is None.
    """
    data_to_plot = OPTIONS["fit.data"]\
        if data_to_plot is None else data_to_plot
    wavelengths = OPTIONS["fit.wavelengths"]
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
            figsize=figsize, facecolor=OPTIONS["plot.color.background"])
    gs = GridSpec(2, nplots, height_ratios=[3, 1])
    axarr = {key: value for key, value in zip(
        data_types, [[fig.add_subplot(gs[j, i], facecolor=OPTIONS["plot.color.background"])
                      for j in range(2)] for i in range(nplots)])}

    plot_datapoints(axarr, axis_ratio, pos_angle,
                    wavelengths, components=components,
                    wavelength_range=wavelength_range,
                    norm=norm, data_to_plot=data_to_plot,
                    colormap=colormap)

    sm = cm.ScalarMappable(cmap=get_colormap(colormap), norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axarr[data_types[-1]])
    cbar.set_ticks(wavelengths.value)
    cbar.set_ticklabels([f"{wavelength:.1f}"
                         for wavelength in wavelengths.value])
    if OPTIONS["plot.color.background"] == "black":
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")
        for spine in cbar.ax.spines.values():
            spine.set_edgecolor("white")
    opposite_color = "white" if OPTIONS["plot.color.background"] == "black"\
        else "black"
    cbar.set_label(label=r"$\lambda$ ($\mathrm{\mu}$m)",
                   color=opposite_color)

    label_color = "lightgray" if OPTIONS["plot.color.background"] == "black"\
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
            lower_ax.set_xlabel(r"Wavelength ($\mathrm{\mu}$m)")
            lower_ax.set_ylabel("Residuals (Jy)")
            upper_ax.tick_params(**tick_settings)
            upper_ax.set_ylabel("Fluxes (Jy)")
            if "flux" in ylimits:
                upper_ax.set_ylim(ylimits["flux"])
            else:
                upper_ax.set_ylim([0, None])
            if not len(axarr) > 1:
                legend = upper_ax.legend(handles=[dot_label, x_label])
                set_legend_color(legend, OPTIONS["plot.color.background"])

        if key in ["vis", "vis2"]:
            lower_ax.set_xlabel(r"$\mathrm{B}_{\mathrm{eff}}$ (M$\lambda$)")

            if key == "vis":
                residual_label = "Residuals (Jy)"
                y_label = "Correlated fluxes (Jy)"
                if "vis" in ylimits:
                    upper_ax.set_ylim(ylimits["vis"])
            else:
                residual_label = "Residuals (Normalized)"
                y_label = "Visibilities (Normalized)"
                if "vis2" in ylimits:
                    upper_ax.set_ylim(ylimits["vis2"])
                else:
                    upper_ax.set_ylim([0, 1])
            lower_ax.set_ylabel(residual_label)
            upper_ax.set_ylabel(y_label)
            upper_ax.tick_params(**tick_settings)
            upper_ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
            if not len(axarr) > 1:
                legend = upper_ax.legend(handles=[dot_label, x_label])
                set_legend_color(legend, OPTIONS["plot.color.background"])

        if key == "t3phi":
            upper_ax.set_ylabel(r"Closure Phases ($^\circ$)")
            lower_ax.set_xlabel(r"$\mathrm{B}_{\mathrm{max}}$ (M$\lambda$)")
            lower_ax.set_ylabel(r"Residuals ($^\circ$)")
            lower_bound = np.min([np.min(value) for value in OPTIONS["data.cphase"]])
            lower_bound += lower_bound*0.25
            upper_bound = np.max([np.max(value) for value in OPTIONS["data.cphase"]])
            upper_bound += upper_bound*0.25
            upper_ax.tick_params(**tick_settings)
            if "t3phi" in ylimits:
                upper_ax.set_ylim(ylimits["t3phi"])
            else:
                upper_ax.set_ylim([lower_bound, upper_bound])
            legend = upper_ax.legend(handles=[dot_label, x_label])
            set_legend_color(legend, OPTIONS["plot.color.background"])

    if title is not None:
        plt.title(title)

    if savefig is not None:
        plt.savefig(savefig, format=Path(savefig).suffix[1:],
                    dpi=OPTIONS["plot.dpi"])
    else:
        plt.show()
    plt.close()


def plot_overview(data_to_plot: Optional[List[str]] = None,
                  colormap: Optional[str] = OPTIONS["plot.color.colormap"],
                  wavelength_range: Optional[List[float]] = None,
                  ylimits: Optional[Dict[str, List[float]]] = {},
                  title: Optional[str] = None,
                  savefig: Optional[Path] = None) -> None:
    """Plots an overview over the total data for baselines [Mlambda].

    Parameters
    ----------
    data_to_plot : list of str, optional
        The data to plot. The default is OPTIONS["fit.data"].
    savefig : pathlib.Path, optional
        The save path. The default is None.
    """
    data_to_plot = OPTIONS["fit.data"]\
        if data_to_plot is None else data_to_plot
    wavelengths = OPTIONS["fit.wavelengths"]
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
    _, axarr = plt.subplots(1, nplots, figsize=figsize,
                            tight_layout=True,
                            facecolor=OPTIONS["plot.color.background"])
    axarr = dict(zip(data_types, axarr.flatten()))

    colormap = get_colormap(colormap)
    hline_color = "gray" if OPTIONS["plot.color.background"] == "white"\
        else "white"

    fluxes, fluxes_err =\
        OPTIONS["data.flux"], OPTIONS["data.flux_err"]

    if "vis" in OPTIONS["fit.data"]:
        vis, vis_err =\
            OPTIONS["data.corr_flux"], OPTIONS["data.corr_flux_err"]
        ucoord, vcoord =\
            OPTIONS["data.corr_flux.ucoord"], OPTIONS["data.corr_flux.vcoord"]
    else:
        vis, vis_err =\
            OPTIONS["data.vis"], OPTIONS["data.vis_err"]
        ucoord, vcoord =\
            OPTIONS["data.vis.ucoord"], OPTIONS["data.vis.vcoord"]

    cphases, cphases_err =\
        OPTIONS["data.cphase"], OPTIONS["data.cphase_err"]
    u123coord = OPTIONS["data.cphase.u123coord"]
    v123coord = OPTIONS["data.cphase.v123coord"]
    errorbar_params = OPTIONS["plot.errorbar"]
    if OPTIONS["plot.color.background"] == "black":
        errorbar_params["markeredgecolor"] = "white"

    for index, wavelength in enumerate(wavelengths):
        if wavelength_range is not None:
            if not (wavelength_range[0] <= wavelength <= wavelength_range[1]):
                continue
        effective_baselines = np.hypot(ucoord[index], vcoord[index])*u.m
        longest_baselines = np.hypot(u123coord[index], v123coord[index]).max(axis=0)*u.m

        effective_baselines_mlambda = effective_baselines/wavelength.value
        longest_baselines_mlambda = longest_baselines/wavelength.value
        color = colormap(norm(wavelength.value))
        errorbar_params["color"] = color

        for key in data_to_plot:
            ax_key = "vis" if key in ["vis", "vis2"] else key
            ax = axarr[ax_key]
            if key == "flux":
                for flux, flux_err in zip(
                        fluxes[index], fluxes_err[index]):
                    ax.errorbar(
                        wavelength.value, flux, flux_err,
                        fmt="o", **errorbar_params)

            if key in ["vis", "vis2"]:
                ax.errorbar(
                    effective_baselines_mlambda.value,
                    vis[index], vis_err[index],
                    fmt="o", **errorbar_params)

            if key == "t3phi":
                ax.errorbar(
                    longest_baselines_mlambda.value,
                    cphases[index], cphases_err[index],
                    fmt="o", **errorbar_params)
                ax.axhline(y=0, color=hline_color, linestyle='--')

    if "flux" in data_to_plot:
        axarr["flux"].set_xticks(wavelengths.value)
        axarr["flux"].set_xticklabels(wavelengths.value, rotation=45)

    errorbar_params["color"] = ""

    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axarr[data_types[-1]])
    cbar.set_ticks(wavelengths.value)
    cbar.set_ticklabels([f"{wavelength:.1f}"
                         for wavelength in wavelengths.value])
    if OPTIONS["plot.color.background"] == "black":
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")
        for spine in cbar.ax.spines.values():
            spine.set_edgecolor("white")
    opposite_color = "white" if OPTIONS["plot.color.background"] == "black"\
        else "black"
    cbar.set_label(label=r"$\lambda$ ($\mathrm{\mu}$m)",
                   color=opposite_color)

    for key in data_to_plot:
        ax_key = "vis" if key in ["vis", "vis2"] else key
        ax = axarr[ax_key]
        set_axes_color(ax, OPTIONS["plot.color.background"])

        if key == "flux":
            ax.set_xlabel(r"$\lambda$ ($\mathrm{\mu}$m)")
            ax.set_ylabel("Fluxes (Jy)")
            if "flux" in ylimits:
                ax.set_ylim(ylimits["flux"])
            else:
                ax.set_ylim([0, None])

        if key == "vis":
            ax.set_xlabel(r"$\mathrm{B}$ (M$\lambda$)")
            ax.set_ylabel("Correlated fluxes (Jy)")
            if "vis" in ylimits:
                ax.set_ylim(ylimits["vis"])
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

        if key == "vis2":
            ax.set_xlabel(r"$\mathrm{B}$ (M$\lambda$)")
            ax.set_ylabel("Visibilities (Normalized)")
            if "vis2" in ylimits:
                ax.set_ylim(ylimits["vis2"])
            else:
                ax.set_ylim([0, 1])
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

        if key == "t3phi":
            ax.set_xlabel(r"$\mathrm{B}_{\mathrm{max}}$ (M$\lambda$)")
            ax.set_ylabel(r"Closure Phases ($^\circ$)")
            lower_bound = np.min([np.min(value) for value in cphases])
            lower_bound += lower_bound*0.25
            upper_bound = np.max([np.max(value) for value in cphases])
            upper_bound += upper_bound*0.25
            if "t3phi" in ylimits:
                ax.set_ylim(ylimits["t3phi"])
            else:
                ax.set_ylim([lower_bound, upper_bound])

    if title is not None:
        plt.title(title)

    if savefig is not None:
        plt.savefig(savefig, format=Path(savefig).suffix[1:],
                    dpi=OPTIONS["plot.dpi"])
    else:
        plt.show()
    plt.close()


def plot_observables(wavelength_range: u.um,
                     components: List[Component],
                     fits_files: List[Path],
                     corr_flux: Optional[bool] = None,
                     save_dir: Optional[Path] = None) -> None:
    """Plots the observables of the model.

    Parameters
    ----------
    wavelength_range : astropy.units.m
    """
    save_dir = Path.cwd() if save_dir is None else save_dir
    wavelengths = np.linspace(wavelength_range[0],
                              wavelength_range[1])

    readouts = list(map(ReadoutFits, fits_files))
    ucoord = np.concatenate([readout.ucoord for readout in readouts])
    vcoord = np.concatenate([readout.vcoord for readout in readouts])
    u123coord = np.hstack([readout.u123coord for readout in readouts])
    v123coord = np.hstack([readout.v123coord for readout in readouts])

    corr_flux = not "vis2" in OPTIONS["fit.data"] if corr_flux is None\
        else corr_flux

    flux = []
    vis = np.empty([len(wavelengths), ucoord.shape[0]])
    cphase = np.empty([len(wavelengths), u123coord.shape[1]])
    for index, wavelength in enumerate(wavelengths):
        stellar_flux = components[0].calculate_stellar_flux(wavelength).value
        tmp_flux, tmp_vis, tmp_cphase = 0, None, None
        for component in components[1:]:
            tmp_flux += component.calculate_flux(wavelength)

            if tmp_vis is None:
                tmp_vis = component.calculate_corr_flux(
                        ucoord, vcoord, wavelength).tolist()
                tmp_cphase = component.calculate_closure_phase(
                        u123coord, v123coord, wavelength).tolist()
            else:
                tmp_vis.extend(component.calculate_corr_flux(
                        ucoord, vcoord, wavelength).tolist())
                tmp_cphase.extend(component.calculate_closure_phase(
                        u123coord, v123coord, wavelength).tolist())

        tmp_flux += stellar_flux
        tmp_vis += stellar_flux

        if not corr_flux:
            tmp_vis /= tmp_flux

        flux.append(tmp_flux)
        vis[index] = tmp_vis
        cphase[index] = tmp_cphase
    flux = np.array(flux)
    _, ax = plt.subplots(tight_layout=True)
    ax.plot(wavelengths, flux)
    ax.set_xlabel(r"$\lambda$ ($\mu$m)")
    ax.set_ylabel("Flux (Jy)")
    plt.savefig(save_dir / "sed.pdf", format="pdf")
    plt.close()

    baseline_dir = save_dir / "baselines"
    baseline_dir.mkdir(exist_ok=True, parents=True)
    vis_dir = baseline_dir / "vis"
    vis_dir.mkdir(exist_ok=True, parents=True)
    cphase_dir = baseline_dir / "cphase"
    cphase_dir.mkdir(exist_ok=True, parents=True)
    for index, (uc, vc) in enumerate(zip(ucoord, vcoord)):
        _, ax = plt.subplots(tight_layout=True)
        baseline, baseline_angle = np.hypot(uc, vc), np.arctan2(uc, vc)*u.rad.to(u.deg)
        ax.plot(wavelengths, vis[:, index],
                label=rf"B={baseline:.2f} m, $\phi$={baseline_angle:.2f}$^\circ$")
        ax.set_xlabel(r"$\lambda$ ($\mu$m)")
        ax.set_ylabel("Visibilities (Normalized)")
        ax.set_ylim([0, 1])
        plt.legend()
        plt.savefig(vis_dir / f"vis_{baseline:.2f}.pdf", format="pdf")
        plt.close()

    for index, baseline in enumerate(np.hypot(u123coord, v123coord).max(axis=0)):
        _, ax = plt.subplots(tight_layout=True)
        ax.plot(wavelengths, cphase[:, index], label=f"B={baseline:.2f} m")
        ax.set_xlabel(r"$\lambda$ ($\mu$m)")
        ax.set_ylabel(r"Closure Phases ($^\circ$)")
        plt.legend()
        plt.savefig(cphase_dir / f"t3phi_{baseline:.2f}.pdf", format="pdf")
        plt.close()
