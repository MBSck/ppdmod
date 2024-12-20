from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import toml
from matplotlib import colormaps as mcm
from matplotlib.colors import ListedColormap


def convert_style_to_colormap(style: str) -> ListedColormap:
    """Converts a style into a colormap."""
    plt.style.use(style)
    colormap = ListedColormap(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    plt.style.use("default")
    return colormap


def get_colormap(colormap: str) -> ListedColormap:
    """Gets the colormap as the matplotlib colormaps or styles."""
    try:
        return mcm.get_cmap(colormap)
    except ValueError:
        return convert_style_to_colormap(colormap)


def get_colorlist(colormap: str, ncolors: int = 10) -> List[str]:
    """Gets the colormap as a list from the matplotlib colormaps."""
    return [get_colormap(colormap)(i) for i in range(ncolors)]


def get_units(dictionary: Dict[str, Any]) -> Dict[str, Any]:
    """Converts the units in a dictionary to astropy units."""
    converted_dictionary = dictionary.copy()
    for value in converted_dictionary.values():
        if "unit" in value:
            if value["unit"] == "one":
                value["unit"] = u.one
            else:
                value["unit"] = u.Unit(value["unit"])

    return converted_dictionary


def load_toml_to_namespace(toml_file: Path):
    """Loads a toml file into a namespace."""
    with open(toml_file, "r") as file:
        data = toml.load(file)["STANDARD_PARAMETERS"]

    return SimpleNamespace(**get_units(data))


STANDARD_PARAMS = load_toml_to_namespace(
    Path(__file__).parent / "config" / "standard_parameters.toml"
)


# NOTE: Data
vis_data = SimpleNamespace(
    value=np.array([]),
    err=np.array([]),
    ucoord=np.array([]).reshape(1, -1),
    vcoord=np.array([]).reshape(1, -1),
)
vis2_data = SimpleNamespace(
    value=np.array([]),
    err=np.array([]),
    ucoord=np.array([]).reshape(1, -1),
    vcoord=np.array([]).reshape(1, -1),
)
t3_data = SimpleNamespace(
    value=np.array([]),
    err=np.array([]),
    u123coord=np.array([]),
    v123coord=np.array([]),
    ucoord=np.array([]).reshape(1, -1),
    vcoord=np.array([]).reshape(1, -1),
    index123=np.array([]),
)
flux_data = SimpleNamespace(value=np.array([]), err=np.array([]))
gravity = SimpleNamespace(index=20)
dtype = SimpleNamespace(complex=np.complex64, real=np.float32)
binning = SimpleNamespace(
    unknown=0.1 * u.um,
    kband=0.1 * u.um,
    hband=0.1 * u.um,
    lband=0.05 * u.um,
    mband=0.05 * u.um,
    lmband=0.05 * u.um,
    nband=0.05 * u.um,
)
interpolation = SimpleNamespace(dim=10, kind="linear", fill_value=None)
data = SimpleNamespace(
    readouts=[],
    bands=[],
    resolutions=[],
    nbaselines=[],
    do_bin=True,
    flux=flux_data,
    vis=vis_data,
    vis2=vis2_data,
    t3=t3_data,
    gravity=gravity,
    binning=binning,
    dtype=dtype,
    interpolation=interpolation,
)

# NOTE: Model
model = SimpleNamespace(
    components=None,
    output="non-normed",
    gridtype="logarithmic",
    modulation=1,
)

# NOTE: Plot
color = SimpleNamespace(
    background="white", colormap="plasma", number=100, list=get_colorlist("plasma", 100)
)
errorbar = SimpleNamespace(
    color=None,
    markeredgecolor="black",
    markeredgewidth=0.2,
    capsize=5,
    capthick=3,
    ecolor="gray",
    zorder=2,
)
scatter = SimpleNamespace(color="", edgecolor="black", linewidths=0.2, zorder=3)
plot = SimpleNamespace(
    dim=4096,
    dpi=300,
    ticks=[1.7, 2.15, 3.2, 4.7, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0],
    color=color,
    errorbar=errorbar,
    scatter=scatter,
)

# NOTE: Weights
weights = SimpleNamespace(flux=1, t3=1, vis=1)
log_f = SimpleNamespace(flux=None, t3=None, vis=None)

# NOTE: Fitting
fit = SimpleNamespace(
    weights=weights,
    log_f=log_f,
    data=["flux", "vis", "t3"],
    wavelengths=None,
    quantiles=[2.5, 50, 97.5],
)

# NOTE: All options
OPTIONS = SimpleNamespace(data=data, model=model, plot=plot, fit=fit)
