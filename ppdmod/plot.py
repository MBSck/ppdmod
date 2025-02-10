import re
from itertools import chain, zip_longest
from pathlib import Path
from typing import Dict, List, Tuple

import astropy.constants as const
import astropy.units as u
import corner
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from dynesty import DynamicNestedSampler, NestedSampler
from dynesty import plotting as dyplot
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from matplotlib.legend import Legend

from .component import FourierComponent
from .fitting import compute_observables, get_best_fit
from .options import OPTIONS, get_colormap
from .utils import (
    angular_to_distance,
    compare_angles,
    compute_effective_baselines,
    distance_to_angular,
    get_band,
    percentile_indices,
)


def get_best_plot_arrangement(nplots):
    """Gets the best plot arrangement for a given number of plots."""
    sqrt_nplots = np.sqrt(nplots)
    cols = int(np.ceil(sqrt_nplots))
    rows = int(np.floor(sqrt_nplots))

    while rows * cols < nplots:
        if cols < rows:
            cols += 1
        else:
            rows += 1

    while (rows - 1) * cols >= nplots:
        rows -= 1

    return rows, cols


def set_axes_color(
    ax: Axes,
    background_color: str,
    set_label: bool = True,
    direction: str = None,
) -> None:
    """Sets all the axes' facecolor."""
    opposite_color = "white" if background_color == "black" else "black"
    ax.set_facecolor(background_color)
    ax.spines["bottom"].set_color(opposite_color)
    ax.spines["top"].set_color(opposite_color)
    ax.spines["right"].set_color(opposite_color)
    ax.spines["left"].set_color(opposite_color)

    if set_label:
        ax.xaxis.label.set_color(opposite_color)
        ax.yaxis.label.set_color(opposite_color)

    ax.tick_params(axis="both", colors=opposite_color, direction=direction)


def set_legend_color(legend: Legend, background_color: str) -> None:
    """Sets the legend's facecolor."""
    opposite_color = "white" if background_color == "black" else "black"
    plt.setp(legend.get_texts(), color=opposite_color)
    legend.get_frame().set_facecolor(background_color)


def plot_components(
    components: List[FourierComponent],
    dim: int,
    pixel_size: u.mas,
    wavelength: u.um,
    norm: float = 0.5,
    save_as_fits: bool = False,
    zoom: float | None = None,
    ax: Axes | None = None,
    cmap: str = "inferno",
    no_text: bool = False,
    savefig: Path | None = None,
) -> Tuple[Axes]:
    """Plots a component."""
    components = [components] if not isinstance(components, list) else components
    image = sum(
        [comp.compute_image(dim, pixel_size, wavelength) for comp in components]
    )[0]

    if any(hasattr(component, "dist") for component in components):
        dist = [component for component in components if hasattr(component, "dist")][
            0
        ].dist()
        pixel_size = angular_to_distance(pixel_size * u.mas, dist).to(u.au).value

    extent = u.Quantity([sign * dim * pixel_size / 2 for sign in [-1, 1, 1, -1]])
    if save_as_fits:
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = (dim / 2, dim / 2)
        wcs.wcs.cdelt = (pixel_size * u.mas.to(u.rad), pixel_size * u.mas.to(u.rad))
        wcs.wcs.crval = (0.0, 0.0)
        wcs.wcs.cunit = (u.rad, u.rad)
        hdu = fits.HDUList([fits.PrimaryHDU(image, header=wcs.to_header())])
        hdu.writeto(savefig, overwrite=True)
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1)

        ax.imshow(image, extent=extent, norm=mcolors.PowerNorm(gamma=norm), cmap=cmap)

        top_ax, right_ax = None, None
        if any(hasattr(component, "dist") for component in components):
            dist = [
                component for component in components if hasattr(component, "dist")
            ][0].dist()

            def convert_to_au(x):
                return angular_to_distance(x * u.mas, dist).to(u.au).value

            def convert_to_mas(x):
                return distance_to_angular(x * u.au, dist).value

            top_ax = ax.secondary_xaxis(
                "top", functions=(convert_to_mas, convert_to_au)
            )
            right_ax = ax.secondary_yaxis(
                "right", functions=(convert_to_mas, convert_to_au)
            )
            top_ax.set_xlabel(r"$\alpha$ (mas)")
            right_ax.set_xlabel(r"$\delta$ (mas)")

        if not no_text:
            ax.text(
                0.18,
                0.95,
                rf"$\lambda$ = {wavelength} " + r"$\mathrm{\mu}$m",
                transform=ax.transAxes,
                fontsize=12,
                color="white",
                ha="center",
            )

        bx = ax.twinx()
        sm = cm.ScalarMappable(cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=bx)

        if zoom is not None:
            ax.set_xlim([zoom, -zoom])
            ax.set_ylim([-zoom, zoom])

        ax.set_xlabel(r"$\alpha$ (au)")
        ax.set_ylabel(r"$\delta$ (au)")

        plt.subplots_adjust(right=0.85)

        if savefig is not None:
            plt.savefig(savefig, format=Path(savefig).suffix[1:], dpi=OPTIONS.plot.dpi)
        else:
            plt.show()

        return ax, top_ax, right_ax, image


def plot_component_mosaic(
    components: List[FourierComponent],
    dim: int,
    pixel_size: u.mas,
    norm: float = 0.5,
    zoom: float = None,
    cols: int = 4,
    cmap: str = "inferno",
    savefig: Path | None = None,
) -> None:
    """Plots a mosaic of components for the different wavelengths."""
    wavelengths = OPTIONS.plot.ticks * u.um
    num_plots = np.array(wavelengths).size
    rows = int(np.ceil(num_plots / cols))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(20, 8),
        gridspec_kw={"hspace": 0, "wspace": 0},
        facecolor=OPTIONS.plot.color.background,
    )

    images = []
    for index, (ax, wavelength) in enumerate(zip(axes.flatten(), wavelengths.value)):
        _, top_ax, right_ax, image = plot_components(
            components,
            dim,
            pixel_size,
            wavelength,
            no_text=True,
            norm=norm,
            zoom=zoom,
            ax=ax,
            cmap=cmap,
        )

        # set_axes_color(ax, "black", set_label=False, direction="in")
        # set_axes_color(top_ax, "black", set_label=False, direction="in")
        # set_axes_color(right_ax, "black", set_label=False, direction="in")

        # if index // cols != 0:
        #     top_ax.tick_params(top=False)
        # if index // cols != (wavelength.size // cols) - 1:
        #     ax.tick_params(bottom=False)
        # if rows:
        # ax.tick_params(labelbottom=False)

        images.append(image)

        ax.text(
            0.18,
            0.95,
            rf"$\lambda$ = {wavelength} $\mathrm{{\mu}}$m",
            transform=ax.transAxes,
            fontsize=12,
            color="white",
            ha="center",
        )

    images = np.array(images)

    [fig.delaxes(ax) for ax in axes.flatten()[num_plots:]]

    plt.tight_layout()

    if savefig is not None:
        plt.savefig(savefig, format=Path(savefig).suffix[1:], dpi=OPTIONS.plot.dpi)
    else:
        plt.show()
    plt.close()


def format_labels(
    labels: List[str], units: List[str | None] = None, split: bool = False
) -> List[str]:
    """Formats the labels in LaTeX.

    Parameters
    ----------
    labels : list of str
        The labels.
    units : list, optional
        The units. The default is None.
    split : bool, optional
        If True, splits into labels, units, and uncertainties. The default is False.

    Returns
    -------
    labels : list of str
        The formatted labels.
    units : list of str, optional
        The formatted units. If split is True
    """
    nice_labels = {
        "rin": {"letter": "R", "indices": [r"\mathrm{in}"]},
        "rout": {"letter": "R", "indices": [r"\mathrm{out}"]},
        "p": {"letter": "p", "indices": []},
        "q": {"letter": "q", "indices": []},
        "rho": {"letter": r"\rho", "indices": []},
        "theta": {"letter": r"\theta", "indices": []},
        "logsigma0": {"letter": r"\Sigma", "indices": ["0"]},
        "sigma0": {"letter": r"\Sigma", "indices": ["0"]},
        "cont_weight": {"letter": "w", "indices": [r"\mathrm{cont}"]},
        "pa": {"letter": r"\theta", "indices": []},
        "cinc": {"letter": r"\cos\left(i\right)", "indices": []},
        "temp0": {"letter": "T", "indices": ["0"]},
        "tempc": {"letter": "T", "indices": [r"\mathrm{c}"]},
    }

    formatted_labels = []
    for label in labels:
        if "-" in label:
            name, index = label.split("-")
        else:
            name, index = label, ""

        if name in nice_labels or name[-1].isdigit():
            if name not in nice_labels and name[-1].isdigit():
                letter = nice_labels[name[:-1]]["letter"]
                indices = [name[-1]]
                if index:
                    indices.append(index)
            else:
                letter = nice_labels[name]["letter"]
                if name in ["temp0", "tempc"]:
                    indices = nice_labels[name]["indices"]
                else:
                    indices = [*nice_labels[name]["indices"]]
                    if index:
                        indices.append(rf"\mathrm{{{index}}}")

            indices = r",\,".join(indices)
            formatted_label = f"{letter}_{{{indices}}}"
            if "log" in label:
                formatted_label = rf"\log_{{10}}\left({formatted_label}\right)"

            formatted_labels.append(f"${formatted_label}$")
        else:
            if "weight" in name:
                name, letter = name.replace("weight", ""), "w"

                indices = []
                if "small" in name:
                    name = name.replace("small", "")
                    indices = [r"\mathrm{small}"]
                elif "large" in name:
                    name = name.replace("large", "")
                    indices = [r"\mathrm{large}"]
                name = name.replace("_", "")
                indices.append(rf"\mathrm{{{name}}}")

                indices = r",\,".join(indices)
                formatted_label = f"{letter}_{{{indices}}}"
                if "log" in label:
                    formatted_label = rf"\log_{{10}}\left({formatted_label}\right)"
            elif "scale" in name:
                formatted_label = rf"w_{{\mathrm{{{name.replace('scale_', '')}}}}}"
            elif "lnf" in name:
                formatted_label = (
                    rf"\ln\left(f\right)_{{\mathrm{{{name.split('_')[0]}}}}}"
                )
            else:
                formatted_label = label

            formatted_labels.append(f"${formatted_label}$")

    if units is not None:
        reformatted_units = []
        for unit in units:
            if unit == u.g / u.cm**2:
                unit = r"\frac{g}{cm^2}"
            elif unit == u.deg:
                unit = r"^{\circ}"
            elif unit == u.pct:
                unit = r"\%"
            reformatted_units.append(unit)

        reformatted_units = [
            rf"$\left(\mathrm{{{str(unit).strip()}}}\right)$" if str(unit) else ""
            for unit in reformatted_units
        ]
        if split:
            return formatted_labels, reformatted_units

        formatted_labels = [
            rf"{label} {unit}"
            for label, unit in zip(formatted_labels, reformatted_units)
        ]
    return formatted_labels


def needs_sci_notation(ax):
    """Checks if scientific notation is needed"""
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    return (
        abs(x_min) <= 1e-3
        or abs(x_max) <= 1e-3
        or abs(y_min) <= 1e-3
        or abs(y_max) <= 1e-3
    )


def get_exponent(num: float) -> int:
    """Gets the exponent of a number for scientific notation"""
    if num == 0:
        raise ValueError("Number must be non-zero")

    exponent_10 = np.floor(np.log10(abs(num)))
    normalized_num = num / (10**exponent_10)
    return np.floor(np.log10(normalized_num) - np.log10(10**exponent_10)).astype(int)


def plot_corner(
    sampler,
    labels: List[str],
    units: List[str] | None = None,
    fontsize: int = 12,
    discard: int = 0,
    savefig: Path | None = None,
    **kwargs,
) -> None:
    """Plots the corner of the posterior spread.

    Parameters
    ----------
    sampler :
    labels : list of str
        The parameter labels.
    units : list of str, optional
    discard : int, optional
    fontsize : int, optional
        The fontsize. The default is 12.
    savefig : pathlib.Path, optional
        The save path. The default is None.
    """
    labels = format_labels(labels, units)
    quantiles = [x / 100 for x in OPTIONS.fit.quantiles]
    if OPTIONS.fit.fitter == "dynesty":
        results = sampler.results
        _, axarr = dyplot.cornerplot(
            results,
            color="blue",
            labels=labels,
            show_titles=True,
            max_n_ticks=3,
            title_quantiles=quantiles,
            quantiles=quantiles,
        )

        theta, uncertainties = get_best_fit(sampler)
        for index, row in enumerate(axarr):
            for ax in row:
                if ax is not None:
                    if needs_sci_notation(ax):
                        if "Sigma" in ax.get_xlabel():
                            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
                            ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

                        if "Sigma" in ax.get_ylabel():
                            ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
                            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
                            ax.yaxis.get_offset_text().set_position((-0.2, 0))

                    title = ax.get_title()
                    if title and np.abs(theta[index]) <= 1e-3:
                        exponent = get_exponent(theta[index])
                        factor = 10.0**exponent
                        formatted_title = (
                            rf"${theta[index] * factor:.2f}_{{-{uncertainties[index][0] * factor:.2f}}}"
                            rf"^{{+{uncertainties[index][1] * factor:.2f}}}\,1\mathrm{{e}}-{exponent}$"
                        )
                        ax.set_title(
                            f"{labels[index]} = {formatted_title}",
                            fontsize=fontsize - 2,
                        )
    else:
        samples = sampler.get_chain(discard=discard, flat=True)
        corner.corner(samples, labels=labels)

    if savefig is not None:
        plt.savefig(savefig, format=Path(savefig).suffix[1:], dpi=OPTIONS.plot.dpi)
    plt.close()


def plot_chains(
    sampler: NestedSampler | DynamicNestedSampler,
    labels: List[str],
    units: List[str] | None = None,
    savefig: Path | None = None,
    **kwargs,
) -> None:
    """Plots the fitter's chains.

    Parameters
    ----------
    sampler : dynesty.NestedSampler or dynesty.DynamicNestedSampler
        The sampler.
    labels : list of str
        The parameter labels.
    units : list of str, optional
    discard : int, optional
    savefig : pathlib.Path, optional
        The save path. The default is None.
    """
    labels = format_labels(labels, units)
    quantiles = [x / 100 for x in OPTIONS.fit.quantiles]
    results = sampler.results
    dyplot.traceplot(
        results,
        labels=labels,
        truths=np.zeros(len(labels)),
        quantiles=quantiles,
        truth_color="black",
        show_titles=True,
        trace_cmap="viridis",
        connect=True,
        connect_highlight=range(5),
    )

    if savefig:
        plt.savefig(savefig, format=Path(savefig).suffix[1:], dpi=OPTIONS.plot.dpi)
    else:
        plt.show()
    plt.close()


class LogNorm(mcolors.Normalize):
    """Gets the log norm."""

    def __init__(self, vmin=None, vmax=None, clip=False):
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_value = np.log1p(value - self.vmin) / np.log1p(self.vmax - self.vmin)
        return np.ma.masked_array(normalized_value, np.isnan(normalized_value))

    def inverse(self, value):
        return np.expm1(value * np.log1p(self.vmax - self.vmin)) + self.vmin


def set_axis_information(
    axarr: Dict[str, List[Axes]],
    key: str,
    ylims: Dict[str, List[float]],
    inclination=None,
) -> Tuple[Axes, Axes]:
    """Sets the axis labels and limits for the different keys."""
    if isinstance(axarr[key], list):
        upper_ax, lower_ax = axarr[key]
        # set_axes_color(lower_ax, OPTIONS.plot.color.background)
    else:
        upper_ax, lower_ax = axarr[key], None

    tick_params = {
        "axis": "x",
        "which": "both",
        "bottom": True,
        "top": False,
        "labelbottom": False,
    }

    if key == "flux":
        xlabel = r"$\lambda$ ($\mathrm{\mu}$m)"
        residual_label = "Residuals (Jy)"
        ylabel = r"$F_\nu$ (Jy)"

    elif key in ["vis", "vis2"]:
        xlabel = r"$\mathrm{B}$ (M$\lambda$)"
        ylim = ylims.get("vis", None)
        if inclination is not None:
            xlabel = r"$\mathrm{B}_\mathrm{eff}$ (M$\lambda$)"

        if key == "vis":
            ylabel = r"$F_{\nu,\,\mathrm{corr}}$ (Jy)"
            residual_label = "Residuals (Jy)"
        else:
            ylabel = "$V^2$ (a.u.)"
            residual_label = "Residuals (a.u.)"
            upper_ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    elif key == "t3":
        xlabel = r"$\mathrm{B}_{\mathrm{max}}$ (M$\lambda$)"
        ylabel = r"$\phi_{\mathrm{cp}}$ ($^\circ$)"
        residual_label = r"Residuals ($^\circ$)"

    upper_ax.tick_params(**tick_params)
    upper_ax.set_ylabel(ylabel)
    if lower_ax is not None:
        lower_ax.set_xlabel(xlabel)
        lower_ax.set_ylabel(residual_label)
    else:
        upper_ax.set_xlabel(xlabel)

    return upper_ax, lower_ax


def plot_data_vs_model(
    axarr,
    wavelengths: np.ndarray,
    value: np.ndarray,
    err: np.ndarray,
    key: str,
    baselines: np.ndarray | None = None,
    model_data: np.ndarray | None = None,
    colormap: str = OPTIONS.plot.color.colormap,
    bands: List[str] | str = "all",
    inclination=None,
    ylims=None,
    norm=None,
):
    """Plots the data versus the model or just the data if not model data given."""
    upper_ax, lower_ax = set_axis_information(axarr, key, ylims, inclination)
    colormap, alpha = get_colormap(colormap), 1 if lower_ax is None else 0.7
    hline_color = "gray" if OPTIONS.plot.color.background == "white" else "white"
    errorbar_params, scatter_params = OPTIONS.plot.errorbar, OPTIONS.plot.scatter
    if OPTIONS.plot.color.background == "black":
        errorbar_params.markeredgecolor = "white"
        scatter_params.edgecolor = "white"

    wavelength_to_bands = np.array(list(map(get_band, wavelengths.value)))
    band_indices = np.where(wavelength_to_bands.astype(bool))[0]
    if bands != "all":
        band_indices = np.where(
            np.any([wavelength_to_bands == band for band in bands], axis=0)
        )

    band_wl = wavelengths[band_indices]
    band_value, band_err = value[band_indices], err[band_indices]
    if model_data is not None:
        band_model_data = np.ma.masked_array(
            model_data[band_indices], mask=band_value.mask
        )

    set_axes_color(upper_ax, OPTIONS.plot.color.background)
    color = colormap(norm(band_wl.value))
    if baselines is None:
        grid = [wl.repeat(band_value.shape[-1]) for wl in band_wl.value]
    else:
        grid = (baselines / band_wl.value[:, np.newaxis])[:, 1:]

    for index, _ in enumerate(band_wl.value):
        errorbar_params.color = scatter_params.color = color[index]
        upper_ax.errorbar(
            grid[index],
            band_value[index],
            band_err[index],
            alpha=alpha,
            fmt="o",
            **vars(errorbar_params),
        )

        if model_data is not None and lower_ax is not None:
            upper_ax.scatter(
                grid[index],
                band_model_data[index],
                marker="X",
                **vars(scatter_params),
            )

            if key == "t3":
                residuals = compare_angles(band_value[index], band_model_data[index])
            else:
                residuals = band_value[index] - band_model_data[index]

            lower_ax.errorbar(
                grid[index],
                residuals,
                band_err[index],
                fmt="o",
                **vars(errorbar_params),
            )
            lower_ax.axhline(y=0, color=hline_color, linestyle="--")

    if key in ["flux", "vis"]:
        ylim = ylims.get(key, [0, None])
    elif key == "vis2":
        ylim = ylims.get(key, [0, 1])
    else:
        value = OPTIONS.data.t3.value
        lower_bound = np.ma.min(value) + np.ma.min(value) * 0.25
        upper_bound = np.ma.max(value) + np.ma.max(value) * 0.25
        ylim = ylims.get("t3", [lower_bound, upper_bound])

    upper_ax.set_ylim(ylim)

    if not len(axarr) > 1:
        label_color = "lightgray" if OPTIONS.plot.color.background == "black" else "k"
        dot_label = mlines.Line2D(
            [],
            [],
            color=label_color,
            marker="o",
            linestyle="None",
            label="Data",
            alpha=0.6,
        )
        x_label = mlines.Line2D(
            [], [], color=label_color, marker="X", linestyle="None", label="Model"
        )
        legend = upper_ax.legend(handles=[dot_label, x_label])
        set_legend_color(legend, OPTIONS.plot.color.background)

    errorbar_params.color = scatter_params.color = None


def plot_fit(
    components: List | None = None,
    data_to_plot: List[str | None] | None = None,
    cmap: str = OPTIONS.plot.color.colormap,
    ylims: Dict[str, List[float]] = {},
    bands: List[str] | str = "all",
    title: str | None = None,
    savefig: Path | None = None,
):
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
    ylimits : dict of list of float, optional
        The ylimits for the individual keys.
    bands : list of str or str, optional
        The bands to be plotted. The default is "all".
    cmap : str, optional
        The colormap.
    title : str, optional
        The title. The default is None.
    savefig : pathlib.Path, optional
        The save path. The default is None.
    """
    data_to_plot = OPTIONS.fit.data if data_to_plot is None else data_to_plot
    flux, t3 = OPTIONS.data.flux, OPTIONS.data.t3
    vis = OPTIONS.data.vis if "vis" in data_to_plot else OPTIONS.data.vis2
    wavelengths = OPTIONS.fit.wavelengths

    norm = LogNorm(vmin=wavelengths[0].value, vmax=wavelengths[-1].value)

    # TODO: Finish this
    # data_to_plot = [key for key in data_to_plot]
    data_types, nplots = [], 0
    for key in data_to_plot:
        if key in ["vis", "vis2"] and "vis" not in data_types:
            data_types.append("vis")
        else:
            data_types.append(key)
        nplots += 1

    model_flux, model_vis, model_t3 = compute_observables(components)
    pos_angle, inclination = components[0].pa(), components[0].cinc()

    figsize = (16, 5) if nplots == 3 else ((12, 5) if nplots == 2 else None)
    fig = plt.figure(figsize=figsize, facecolor=OPTIONS.plot.color.background)
    gs = GridSpec(2, nplots, height_ratios=[3, 1])
    axarr = {
        key: value
        for key, value in zip(
            data_types,
            [
                [
                    fig.add_subplot(gs[j, i], facecolor=OPTIONS.plot.color.background)
                    for j in range(2)
                ]
                for i in range(nplots)
            ],
        )
    }

    plot_kwargs = {"norm": norm, "colormap": cmap}
    if "flux" in data_to_plot:
        plot_data_vs_model(
            axarr,
            wavelengths,
            flux.value,
            flux.err,
            "flux",
            ylims=ylims,
            bands=bands,
            model_data=model_flux,
            inclination=inclination,
            **plot_kwargs,
        )

    if "vis" in data_to_plot or "vis2" in data_to_plot:
        effective_baselines, _ = compute_effective_baselines(
            vis.ucoord, vis.vcoord, inclination, pos_angle
        )
        plot_data_vs_model(
            axarr,
            wavelengths,
            vis.value,
            vis.err,
            "vis" if "vis" in data_to_plot else "vis2",
            ylims=ylims,
            bands=bands,
            baselines=effective_baselines,
            model_data=model_vis,
            inclination=inclination,
            **plot_kwargs,
        )

    if "t3" in data_to_plot:
        longest_baselines, _ = compute_effective_baselines(
            t3.u123coord, t3.v123coord, inclination, pos_angle, longest=True
        )

        plot_data_vs_model(
            axarr,
            wavelengths,
            t3.value,
            t3.err,
            "t3",
            ylims=ylims,
            bands=bands,
            baselines=longest_baselines,
            model_data=model_t3,
            inclination=inclination,
            **plot_kwargs,
        )

    sm = cm.ScalarMappable(cmap=get_colormap(cmap), norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axarr[data_types[-1]])
    cbar.set_ticks(OPTIONS.plot.ticks)
    cbar.set_ticklabels([f"{wavelength:.1f}" for wavelength in OPTIONS.plot.ticks])

    if OPTIONS.plot.color.background == "black":
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")
        for spine in cbar.ax.spines.values():
            spine.set_edgecolor("white")
    opposite_color = "white" if OPTIONS.plot.color.background == "black" else "black"
    cbar.set_label(label=r"$\lambda$ ($\mathrm{\mu}$m)", color=opposite_color)

    if title is not None:
        plt.title(title)

    if savefig is not None:
        plt.savefig(savefig, format=Path(savefig).suffix[1:], dpi=OPTIONS.plot.dpi)
    else:
        plt.show()
    plt.close()


def plot_overview(
    data_to_plot: List[str | None] = None,
    colormap: str = OPTIONS.plot.color.colormap,
    ylims: Dict[str, List[float]] = {},
    title: str | None = None,
    raxis: bool = False,
    inclination: float | None = None,
    pos_angle: float | None = None,
    bands: List[str] | str = "all",
    savefig: Path | None = None,
) -> None:
    """Plots an overview over the total data for baselines [Mlambda].

    Parameters
    ----------
    data_to_plot : list of str, optional
        The data to plot. The default is OPTIONS.fit.data.
    savefig : pathlib.Path, optional
        The save path. The default is None.
    """
    data_to_plot = OPTIONS.fit.data if data_to_plot is None else data_to_plot
    wavelengths = OPTIONS.fit.wavelengths
    norm = LogNorm(vmin=wavelengths[0].value, vmax=wavelengths[-1].value)

    data_types, nplots = [], 0
    for key in data_to_plot:
        if key in ["vis", "vis2"] and "vis" not in data_types:
            data_types.append("vis")
        else:
            data_types.append(key)
        nplots += 1

    figsize = (15, 5) if nplots == 3 else ((12, 5) if nplots == 2 else None)
    fig, axarr = plt.subplots(
        1,
        nplots,
        figsize=figsize,
        tight_layout=True,
        facecolor=OPTIONS.plot.color.background,
    )
    axarr = axarr.flatten() if isinstance(axarr, np.ndarray) else [axarr]
    axarr = dict(zip(data_types, axarr))

    flux, t3 = OPTIONS.data.flux, OPTIONS.data.t3
    vis = OPTIONS.data.vis if "vis" in OPTIONS.fit.data else OPTIONS.data.vis2

    errorbar_params = OPTIONS.plot.errorbar
    if OPTIONS.plot.color.background == "black":
        errorbar_params.markeredgecolor = "white"

    plot_kwargs = {"norm": norm, "colormap": colormap}
    if "flux" in data_to_plot:
        plot_data_vs_model(
            axarr,
            wavelengths,
            flux.value,
            flux.err,
            "flux",
            ylims=ylims,
            bands=bands,
            inclination=inclination,
            **plot_kwargs,
        )

    if "vis" in data_to_plot or "vis2" in data_to_plot:
        effective_baselines, _ = compute_effective_baselines(
            vis.ucoord, vis.vcoord, inclination, pos_angle
        )
        plot_data_vs_model(
            axarr,
            wavelengths,
            vis.value,
            vis.err,
            "vis" if "vis" in data_to_plot else "vis2",
            ylims=ylims,
            bands=bands,
            baselines=effective_baselines,
            inclination=inclination,
            **plot_kwargs,
        )

    if "t3" in data_to_plot:
        longest_baselines, _ = compute_effective_baselines(
            t3.u123coord, t3.v123coord, inclination, pos_angle, longest=True
        )

        plot_data_vs_model(
            axarr,
            wavelengths,
            t3.value,
            t3.err,
            "t3",
            ylims=ylims,
            bands=bands,
            baselines=longest_baselines,
            inclination=inclination,
            **plot_kwargs,
        )

    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axarr[data_types[-1]])

    # TODO: Set the ticks, but make it so that it is flexible for the band
    cbar.set_ticks(OPTIONS.plot.ticks)
    cbar.set_ticklabels([f"{wavelength:.1f}" for wavelength in OPTIONS.plot.ticks])

    if OPTIONS.plot.color.background == "black":
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")
        for spine in cbar.ax.spines.values():
            spine.set_edgecolor("white")
    opposite_color = "white" if OPTIONS.plot.color.background == "black" else "black"
    cbar.set_label(label=r"$\lambda$ ($\mathrm{\mu}$m)", color=opposite_color)

    if title is not None:
        plt.title(title)

    if raxis:
        return fig, axarr

    if savefig is not None:
        plt.savefig(savefig, format=Path(savefig).suffix[1:], dpi=OPTIONS.plot.dpi)
    else:
        plt.show()
    plt.close()


# TODO: Make colorscale permanent -> Implement colormap
def plot_sed(
    wavelength_range: u.um,
    components: List[FourierComponent | None] = None,
    scaling: str = "nu",
    no_model: bool = False,
    ax: plt.Axes | None = None,
    savefig: Path | None = None,
):
    """Plots the observables of the model.

    Parameters
    ----------
    wavelength_range : astropy.units.m
    scaling : str, optional
        The scaling of the SED. "nu" for the flux to be
        in Jy times Hz. If "lambda" the flux is in Jy times m.
        If "none" the flux is in Jy.
        The default is "nu".
    """
    color = OPTIONS.plot.color
    save_dir = Path.cwd() if save_dir is None else save_dir
    wavelength = np.linspace(wavelength_range[0], wavelength_range[1], OPTIONS.plot.dim)

    if not no_model:
        wavelength = OPTIONS.fit.wavelengths if wavelength is None else wavelength
        components = [comp for comp in components if comp.name != "Point Source"]
        flux = np.sum([comp.compute_flux(wavelength) for comp in components], axis=0)
        if flux.size > 0:
            flux = np.tile(flux, (len(OPTIONS.data.readouts))).real

    if ax is None:
        fig = plt.figure(facecolor=color.background, tight_layout=True)
        ax = plt.axes(facecolor=color.background)
        set_axes_color(ax, color.background)
    else:
        fig = None

    if len(OPTIONS.data.readouts) > 1:
        names = [
            re.findall(r"(\d{4}-\d{2}-\d{2})", readout.fits_file.name)[0]
            for readout in OPTIONS.data.readouts
        ]
    else:
        names = [OPTIONS.data.readouts[0].fits_file.name]

    cmap = plt.get_cmap(color.colormap)
    norm = mcolors.LogNorm(vmin=1, vmax=len(set(names)))
    colors = [cmap(norm(i)) for i in range(1, len(set(names)) + 1)]
    date_to_color = {date: color for date, color in zip(set(names), colors)}
    sorted_readouts = np.array(OPTIONS.data.readouts.copy())[np.argsort(names)].tolist()

    values = []
    for name, readout in zip(np.sort(names), sorted_readouts):
        if readout.flux.value.size == 0:
            continue

        readout_wavelength = readout.wavelength.value
        readout_flux, readout_err = (
            readout.flux.value.flatten(),
            readout.flux.err.flatten(),
        )
        readout_err_percentage = readout_err / readout_flux

        if scaling == "nu":
            readout_flux = (readout_flux * u.Jy).to(u.W / u.m**2 / u.Hz)
            readout_flux = (
                readout_flux
                * (const.c / ((readout_wavelength * u.um).to(u.m))).to(u.Hz)
            ).value

        readout_err = readout_err_percentage * readout_flux
        lower_err, upper_err = readout_flux - readout_err, readout_flux + readout_err
        if "HAW" in readout.fits_file.name:
            indices_high = np.where(
                (readout_wavelength >= 4.55) & (readout_wavelength <= 4.9)
            )
            indices_low = np.where(
                (readout_wavelength >= 3.1) & (readout_wavelength <= 3.9)
            )
            for indices in [indices_high, indices_low]:
                line = ax.plot(
                    readout_wavelength[indices],
                    readout_flux[indices],
                    color=date_to_color[name],
                )
                ax.fill_between(
                    readout_wavelength[indices],
                    lower_err[indices],
                    upper_err[indices],
                    color=line[0].get_color(),
                    alpha=0.5,
                )
            value_indices = np.hstack([indices_high, indices_low])
            lim_values = readout_flux[value_indices].flatten()
        else:
            line = ax.plot(readout_wavelength, readout_flux, color=date_to_color[name])
            ax.fill_between(
                readout_wavelength,
                lower_err,
                upper_err,
                color=line[0].get_color(),
                alpha=0.5,
            )
            lim_values = readout_flux
        values.append(lim_values)

    flux_label = r"$F_{\nu}$ (Jy)"
    if not no_model:
        flux = flux[:, 0]
        if scaling == "nu":
            flux = (flux * u.Jy).to(u.W / u.m**2 / u.Hz)
            flux = (flux * (const.c / (wavelength.to(u.m))).to(u.Hz)).value
            flux_label = r"$\nu F_{\nu}$ (W m$^{-2}$)"

    if not no_model:
        ax.plot(wavelength, flux, label="Model", color="red")
        values.append(flux)

    if fig is not None:
        ax.set_xlabel(r"$\lambda$ ($\mathrm{\mu}$m)")
        ax.set_ylabel(flux_label)
        ax.legend()

        max_value = np.concatenate(values).max()
        ax.set_ylim([0, max_value + 0.1 * max_value])

        if savefig is not None:
            plt.savefig(savefig, format=Path(savefig).suffix[1:], dpi=OPTIONS.plot.dpi)
        plt.close()


def plot_product(
    points,
    product,
    xlabel,
    ylabel,
    save_path=None,
    ax=None,
    colorbar=False,
    cmap: str = OPTIONS.plot.color.colormap,
    scale=None,
    label=None,
):
    norm = None
    if label is not None:
        if isinstance(label, (np.ndarray, u.Quantity)):
            norm = mcolors.Normalize(vmin=label[0].value, vmax=label[-1].value)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if product.ndim > 1:
        for lb, prod in zip(label, product):
            color = None
            if norm is not None:
                colormap = get_colormap(cmap)
                color = colormap(norm(lb.value))
            ax.plot(points, prod, label=lb, color=color)
        if not colorbar:
            ax.legend()
    else:
        ax.plot(points, product, label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if scale == "log":
        ax.set_yscale("log")
    elif scale == "loglog":
        ax.set_yscale("log")
        ax.set_xscale("log")
    elif scale == "sci":
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    if colorbar:
        sm = cm.ScalarMappable(cmap=get_colormap(cmap), norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_ticks(OPTIONS.plot.ticks)
        cbar.set_ticklabels([f"{wavelength:.1f}" for wavelength in OPTIONS.plot.ticks])
        cbar.set_label(label=r"$\lambda$ ($\mathrm{\mu}$m)")

    if save_path is not None:
        fig.savefig(save_path, format=Path(save_path).suffix[1:])
        plt.close(fig)


# TODO: Clean and split this function into multiple ones
def plot_intermediate_products(
    dim: int,
    wavelength: u.Quantity[u.um],
    components: List[FourierComponent],
    component_labels: List[str],
    save_dir: Path | None = None,
) -> None:
    """Plots the intermediate products of the model (temperature, density, etc.)."""
    wavelength = (
        u.Quantity(wavelength, u.um)
        if wavelength is not None
        else OPTIONS.fit.wavelengths
    )
    wavelengths = np.linspace(wavelength[0], wavelength[-1], dim)
    component_labels = [
        " ".join(map(str.title, label.split("_"))) for label in component_labels
    ]
    radii, surface_density, optical_depth = [], [], []
    fluxes, temperature, emissivity, intensity = [], [], [], []

    _, ax = plt.subplots(figsize=(5, 5))
    for label, component in zip(component_labels, components):
        component.dim.value = dim
        flux = component.compute_flux(wavelengths).squeeze()
        plot_product(
            wavelengths,
            flux,
            r"$\lambda$ ($\mathrm{\mu}$m)",
            r"$F_{\nu}$ (Jy)",
            scale="log",
            ax=ax,
            label=label,
        )
        fluxes.append(flux)

        if component.name in ["Star", "Point"]:
            continue

        radius = component.compute_internal_grid()
        radii.append(radius)

        temperature.append(component.compute_temperature(radius))
        surface_density.append(component.compute_surface_density(radius))
        optical_depth.append(
            component.compute_optical_depth(radius, wavelength[:, np.newaxis])
        )
        emissivity.append(
            component.compute_emissivity(radius, wavelength[:, np.newaxis])
        )
        intensity.append(component.compute_intensity(radius, wavelength[:, np.newaxis]))

    temperature = u.Quantity(temperature)
    surface_density = u.Quantity(surface_density)
    optical_depth = u.Quantity(optical_depth)
    emissivity = u.Quantity(emissivity)
    intensity = u.Quantity(intensity)

    total_flux = sum(fluxes)
    ax.plot(wavelengths, total_flux, label="Total")
    ax.set_yscale("log")
    ax.set_ylim([1e-1, None])
    ax.legend()
    plt.savefig(save_dir / "fluxes.png", format="png")
    plt.close()

    _, ax = plt.subplots(figsize=(5, 5))
    for label, flux_ratio in zip(component_labels, fluxes / total_flux * 100):
        plot_product(
            wavelengths,
            flux_ratio,
            r"$\lambda$ ($\mathrm{\mu}$m)",
            r"$F_{\nu}$ / $F_{\nu,\,\mathrm{tot}}$ (%)",
            ax=ax,
            label=label,
        )

    ax.legend()
    ax.set_ylim([0, 100])
    plt.savefig(save_dir / "flux_ratios.png", format="png")
    plt.close()

    radii_bounds = [
        (prev[-1], current[0]) for prev, current in zip(radii[:-1], radii[1:])
    ]
    fill_radii = [np.linspace(lower, upper, dim) for lower, upper in radii_bounds]
    merged_radii = list(chain.from_iterable(zip_longest(radii, fill_radii)))[:-1]
    merged_radii = u.Quantity(np.concatenate(merged_radii, axis=0))
    fill_zeros = np.zeros((len(fill_radii), wavelength.size, dim))

    # TODO: Make it so that the temperatures are somehow continous in the plot? (Maybe check for self.temps in the models?)
    # or interpolate smoothly somehow (see the one youtube video?) :D
    temperature = u.Quantity(
        list(chain.from_iterable(zip_longest(temperature, fill_zeros[:, 0, :] * u.K)))[
            :-1
        ]
    )
    temperature = np.concatenate(temperature, axis=0)
    surface_density = u.Quantity(
        list(
            chain.from_iterable(
                zip_longest(surface_density, fill_zeros[:, 0, :] * u.g / u.cm**2)
            )
        )[:-1]
    )
    surface_density = np.concatenate(surface_density, axis=0)
    optical_depth = u.Quantity(
        list(chain.from_iterable(zip_longest(optical_depth, fill_zeros)))[:-1]
    )
    optical_depth = np.hstack(optical_depth)
    emissivity = u.Quantity(
        list(chain.from_iterable(zip_longest(emissivity, fill_zeros)))[:-1]
    )
    emissivity = np.hstack(emissivity)
    intensity = u.Quantity(
        list(
            chain.from_iterable(
                zip_longest(intensity, fill_zeros * u.erg / u.cm**2 / u.s / u.Hz / u.sr)
            )
        )[:-1]
    )
    intensity = np.hstack(intensity)
    intensity = intensity.to(u.W / u.m**2 / u.Hz / u.sr)
    merged_radii_mas = distance_to_angular(merged_radii, components[-1].dist())

    # TODO: Code this in a better manner
    wls = [1.7, 2.15, 3.4, 8, 11.3, 13] * u.um
    cumulative_intensity = (
        np.zeros((wls.size, merged_radii_mas.size))
        * u.erg
        / u.s
        / u.Hz
        / u.cm**2
        / u.sr
    )
    for index, wl in enumerate(wls):
        tmp_intensity = [
            component.compute_intensity(radius, wl)
            for radius, component in zip(radii, components[1:])
        ]
        tmp_intensity = u.Quantity(
            list(
                chain.from_iterable(
                    zip_longest(
                        tmp_intensity,
                        fill_zeros[0, 0][np.newaxis, :]
                        * u.erg
                        / u.cm**2
                        / u.s
                        / u.Hz
                        / u.sr,
                    )
                )
            )[:-1]
        )
        cumulative_intensity[index, :] = np.hstack(tmp_intensity)

    cumulative_intensity = cumulative_intensity.to(
        u.erg / u.s / u.Hz / u.cm**2 / u.mas**2
    )
    cumulative_total_flux = (
        2
        * np.pi
        * components[-1].cinc()
        * np.trapz(merged_radii_mas * cumulative_intensity, merged_radii_mas).to(u.Jy)[
            :, np.newaxis
        ]
    )

    cumulative_flux = np.zeros((wls.size, merged_radii.size)) * u.Jy
    for index, _ in enumerate(merged_radii):
        cumulative_flux[:, index] = (
            2
            * np.pi
            * components[-1].cinc()
            * np.trapz(
                merged_radii_mas[:index] * cumulative_intensity[:, :index],
                merged_radii_mas[:index],
            ).to(u.Jy)
        )
    cumulative_flux_ratio = cumulative_flux / cumulative_total_flux
    plot_product(
        merged_radii.value,
        cumulative_flux_ratio.value,
        "$R$ (AU)",
        r"$F_{\nu}\left(r\right)/F_{\nu,\,\mathrm{{tot}}}$ (a.u.)",
        label=wls,
        save_path=save_dir / "cumulative_flux_ratio.png",
    )

    plot_product(
        merged_radii.value,
        temperature.value,
        "$R$ (AU)",
        "$T$ (K)",
        save_path=save_dir / "temperature.png",
    )
    plot_product(
        merged_radii.value,
        surface_density.value,
        "$R$ (au)",
        r"$\Sigma$ (g cm$^{-2}$)",
        save_path=save_dir / "surface_density.png",
        scale="sci",
    )
    plot_product(
        merged_radii.value,
        optical_depth.value,
        "$R$ (AU)",
        r"$\tau_{\nu}$",
        save_path=save_dir / "optical_depths.png",
        scale="log",
        colorbar=True,
        label=wavelength,
    )
    # plot_product(merged_radii.value, emissivities.value,
    #              "$R$ (AU)", r"$\epsilon_{\nu}$",
    #              save_path=save_dir / "emissivities.png",
    #              label=wavelength)
    # plot_product(merged_radii.value, brightnesses.value,
    #              "$R$ (AU)", r"$I_{\nu}$ (W m$^{-2}$ Hz$^{-1}$ sr$^{-1}$)",
    #              save_path=save_dir / "brightnesses.png",
    #              scale="log", label=wavelength)
