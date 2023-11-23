from datetime import datetime
from typing import Optional, Any, Dict, List
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
from astropy.table import Table
from astropy.wcs import WCS
from matplotlib import colormaps as mcm
from matplotlib.gridspec import GridSpec

from .component import Component
from .mcmc import calculate_observables
from .model import Model
from .options import OPTIONS
from .utils import calculate_effective_baselines, restrict_phase

matplotlib.use('Agg')

HEADER_DICT = {
        "BUNIT": ("Jy", "Unit of original pixel value"),
        "BTYPE": ("Brightness", "Type of original pixel value"),
        "EXTEND": (True, "EXTEND"),
        "COMMENT": "Best fit model image per wavelength",
        }


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
    samples = sampler.get_chain(discard=discard, flat=True)
    corner.corner(samples, show_titles=True,
                  labels=labels, plot_datapoints=True,
                  quantiles=[0.16, 0.5, 0.84])

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


# TODO: Add components to this as well as the parameters in a sub HDULIST.
def save_fits(dim: int, pixel_size: u.mas, distance: u.pc,
              wavelengths: List[u.Quantity[u.um]],
              components: List[Component],
              component_labels: List[str],
              opacities: List[np.ndarray] = None,
              savefits: Optional[Path] = None,
              options: Optional[Dict[str, Any]] = None,
              object_name: Optional[str] = None) -> None:
    """Saves a (.fits)-file of the model with all the information on the parameter space."""
    pixel_size = pixel_size if isinstance(pixel_size, u.Quantity) else pixel_size*u.mas
    wavelengths = wavelengths if isinstance(wavelengths, u.Quantity) else wavelengths*u.um
    distance = distance if isinstance(distance, u.Quantity) else distance*u.pc

    # TODO: Make this for the numerical case as well.
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
                if not "thickness" in data:
                    data["thickness"] = []
                if not "brightness" in data:
                    data["brightness"] = []
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
            if column.name in ["wavelength", "kappa_abs", "kappa_cont"]:
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


def plot_datapoints(
        axarr, axis_ratio: u.one,
        pos_angle: u.deg, wavelengths: u.um,
        components: Optional[List] = None,
        data_to_plot: Optional[List[str]] = OPTIONS["fit.data"],
        norm: Optional[np.ndarray] = None,
        colormap: Optional[str] = None) -> None:
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
    colormap = mcm.get_cmap(colormap)
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

    for index, wavelength in enumerate(wavelengths):
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

        if "vis" in data_to_plot or "vis2" in data_to_plot:
            upper_ax_vis, lower_ax_vis = axarr["vis"]
            if "flux" in data_to_plot:
                upper_ax_vis.errorbar(
                    np.array([0]), fluxes[index],
                    fluxes_err[index], color=color,
                    markeredgecolor="black", fmt="o", alpha=0.6)
                upper_ax_vis.scatter(
                    np.array([0]), flux_model,
                    edgecolor="black", marker="X", color=color)
                lower_ax_vis.scatter(np.array([0]),
                                     fluxes[index]-flux_model,
                                     edgecolor="black",
                                     color=color, marker="o")
            upper_ax_vis.errorbar(
                effective_baselines_mlambda.value,
                vis[index], vis_err[index], markeredgecolor="black",
                color=color, fmt="o", alpha=0.6)
            upper_ax_vis.scatter(
                effective_baselines_mlambda.value, vis_model,
                edgecolor="black", color=color, marker="X")
            lower_ax_vis.axhline(y=0, color="gray", linestyle='--')
            lower_ax_vis.scatter(effective_baselines_mlambda.value,
                                 vis[index]-vis_model,
                                 edgecolor="black",
                                 color=color, marker="o")
            lower_ax_vis.axhline(y=0, color="gray", linestyle='--')

        if "t3phi" in data_to_plot:
            upper_ax_cphase, lower_ax_cphase = axarr["t3phi"]
            upper_ax_cphase.errorbar(
                longest_baselines_mlambda.value,
                cphases[index], cphases_err[index],
                markeredgecolor="black",
                color=color, fmt="o", alpha=0.6)
            upper_ax_cphase.scatter(
                longest_baselines_mlambda.value, cphase_model,
                edgecolor="black", color=color, marker="X")
            upper_ax_cphase.axhline(y=0, color="gray", linestyle='--')
            lower_ax_cphase.scatter(longest_baselines_mlambda.value,
                                    restrict_phase(cphases[index]-cphase_model),
                                    edgecolor="black",
                                    color=color, marker="o")
            lower_ax_cphase.axhline(y=0, color="gray", linestyle='--')


def plot_fit(axis_ratio: u.one, pos_angle: u.deg,
             components: Optional[List] = None,
             data_to_plot: Optional[List[str]] = None,
             colormap: Optional[str] = "tab20",
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
        The colormap. The default is "tab20".
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
    if ("vis" in data_to_plot or "vis2" in data_to_plot)\
            and "vis" not in data_types:
        nplots += 1
        data_types.append("vis")

    if "t3phi" in data_to_plot:
        nplots += 1
        data_types.append("t3phi")

    figsize = (12, 5) if nplots == 2 else None
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(nplots, 2, height_ratios=[3, 1])
    axarr = {key: value for key, value in zip(
        data_types, [[fig.add_subplot(gs[j, i]) for j in range(2)]
                     for i in range(nplots)])}

    plot_datapoints(axarr, axis_ratio, pos_angle,
                    wavelengths, components=components,
                    norm=norm, data_to_plot=data_to_plot,
                    colormap=colormap)

    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axarr[data_types[-1]],
                        label="Wavelength (micron)")
    cbar.set_ticks(wavelengths.value)
    cbar.set_ticklabels([f"{wavelength:.1f}" for wavelength in wavelengths.value])

    dot_label = mlines.Line2D([], [], color="k", marker="o",
                              linestyle="None", label="Data", alpha=0.6)
    x_label = mlines.Line2D([], [], color="k", marker="X",
                            linestyle="None", label="Model")

    if "vis" in data_types or "vis2" in data_types:
        upper_ax_vis, lower_ax_vis = axarr["vis"]
        lower_ax_vis.set_xlabel(r"$\mathrm{B}_{\mathrm{eff}}/\lambda$ (M$\lambda$)")

        if "vis" in data_types:
            residual_label = "Residuals (Jy)"
            y_label = "Correlated fluxes (Jy)"
        else:
            residual_label = "Residuals (a.u.)"
            y_label = "Visibilities (a.u.)"
            upper_ax_vis.set_ylim([0, 1])
        lower_ax_vis.set_ylabel(residual_label)
        upper_ax_vis.set_ylabel(y_label)
        upper_ax_vis.xaxis.set_visible(False)
        if not len(axarr) > 1:
            upper_ax_vis.legend(handles=[dot_label, x_label])

    if "t3phi" in data_types:
        upper_ax_cphase, lower_ax_cphase = axarr["t3phi"]
        upper_ax_cphase.set_ylabel(r"Closure Phases ($^\circ$)")
        lower_ax_cphase.set_xlabel(r"$\mathrm{B}_{\mathrm{max}}/\lambda$ (M$\lambda$)")
        lower_ax_cphase.set_ylabel(r"Residuals ($^\circ$)")
        lower_bound = np.min([np.min(value) for value in OPTIONS["data.cphase"]])
        lower_bound += lower_bound*0.25
        upper_bound = np.max([np.max(value) for value in OPTIONS["data.cphase"]])
        upper_bound += upper_bound*0.25
        upper_ax_cphase.xaxis.set_visible(False)
        upper_ax_cphase.set_ylim([lower_bound, upper_bound])
        upper_ax_cphase.legend(handles=[dot_label, x_label])

    if title:
        plt.title(title)

    if savefig:
        plt.savefig(savefig, format="pdf")
    else:
        plt.show()
    plt.close()


def plot_overview(data_to_plot: Optional[List[str]] = None,
                  colormap: Optional[str] = "tab20",
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
    if ("vis" in data_to_plot or "vis2" in data_to_plot)\
            and "vis" not in data_types:
        nplots += 1
        data_types.append("vis")

    if "t3phi" in data_to_plot:
        nplots += 1
        data_types.append("t3phi")

    figsize = (12, 5) if nplots == 2 else None
    _, axarr = plt.subplots(1, nplots, figsize=figsize)
    axarr = dict(zip(data_types, axarr.flatten()))

    colormap = mcm.get_cmap(colormap)
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

    for index, wavelength in enumerate(wavelengths):
        effective_baselines = np.hypot(ucoord[index], vcoord[index])*u.m
        longest_baselines = np.hypot(u123coord[index], v123coord[index]).max(axis=0)*u.m

        effective_baselines_mlambda = effective_baselines/wavelength.value
        longest_baselines_mlambda = longest_baselines/wavelength.value
        color = colormap(norm(wavelength.value))

        # TODO: Add total flux here
        if "vis" in data_to_plot or "vis2" in data_to_plot:
            ax = axarr["vis"]
            ax.errorbar(
                effective_baselines_mlambda.value,
                vis[index], vis_err[index],
                color=color, fmt="o", alpha=0.6)

        if "t3phi" in axarr:
            cx = axarr["t3phi"]
            cx.errorbar(
                longest_baselines_mlambda.value,
                cphases[index], cphases_err[index],
                color=color, fmt="o", alpha=0.6)
            cx.axhline(y=0, color="gray", linestyle='--')

    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axarr[data_types[-1]],
                        label="Wavelength (micron)")
    cbar.set_ticks(wavelengths.value)
    cbar.set_ticklabels([f"{wavelength:.1f}"
                         for wavelength in wavelengths.value])

    if "vis" in data_to_plot:
        ax = axarr["vis"]
        ax.set_xlabel(r"$\mathrm{B}k/\lambda$ (M$\lambda$)")
        ax.set_ylabel("Correlated fluxes (Jy)")

    if "vis2" in data_to_plot:
        bx = axarr["vis"]
        bx.set_xlabel(r"$\mathrm{B}/\lambda$ (M$\lambda$)")
        bx.set_ylabel("Visibilities (a.u.)")
        bx.set_ylim([0, 1])

    if "t3phi" in data_to_plot:
        cx = axarr["t3phi"]
        cx.set_xlabel(r"$\mathrm{B}_{\mathrm{max}}/\lambda$ (M$\lambda$)")
        cx.set_ylabel(r"Closure Phases ($^\circ$)")
        lower_bound = np.min([np.min(value) for value in cphases])
        lower_bound += lower_bound*0.25
        upper_bound = np.max([np.max(value) for value in cphases])
        upper_bound += upper_bound*0.25
        cx.set_ylim([lower_bound, upper_bound])

    if title:
        plt.title(title)

    if savefig:
        plt.savefig(savefig, format="pdf")
    else:
        plt.show()
    plt.close()
