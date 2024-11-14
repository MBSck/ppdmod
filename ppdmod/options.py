from pathlib import Path
from types import SimpleNamespace
from typing import List

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
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


# NOTE: A list of standard parameters to be used when defining new components.
STANDARD_PARAMETERS = SimpleNamespace(
    x={
        "name": "x",
        "shortname": "x",
        "value": 0,
        "description": "The x position",
        "min": -30,
        "max": 30,
        "unit": u.mas,
        "free": False,
    },
    y={
        "name": "y",
        "shortname": "y",
        "value": 0,
        "description": "The y position",
        "min": -30,
        "max": 30,
        "unit": u.mas,
        "free": False,
    },
    f={
        "name": "flux",
        "shortname": "f",
        "value": None,
        "description": "The flux",
        "min": 0,
        "max": 10,
        "smooth": True,
        "unit": u.Jy,
        "free": False,
    },
    fr={
        "name": "flux_ratio",
        "shortname": "fr",
        "value": 1,
        "description": "The flux ratio",
        "min": 0,
        "max": 1,
        "unit": u.one,
        "free": False,
    },
    inc={
        "name": "inclination",
        "shortname": "inc",
        "value": 1,
        "min": 0,
        "max": 1,
        "unit": u.one,
        "description": "The cosine of the inclination",
    },
    pa={
        "name": "position_angle",
        "shortname": "pa",
        "value": 0,
        "min": 0,
        "max": 180,
        "unit": u.deg,
        "description": "The major-axis position angle",
    },
    pixel_size={
        "name": "pixel_size",
        "shortname": "pix_size",
        "value": 1,
        "min": 0,
        "max": 100,
        "description": "The pixel size",
        "unit": u.mas,
        "free": False,
    },
    dim={
        "name": "dim",
        "shortname": "dim",
        "value": 128,
        "description": "The pixel dimension",
        "unit": u.one,
        "dtype": int,
        "free": False,
    },
    exp={
        "name": "exponent",
        "shortname": "exp",
        "value": 1,
        "min": 0,
        "max": 1,
        "unit": u.one,
        "description": "An exponent",
        "free": True,
    },
    wl={
        "name": "wl",
        "shortname": "wl",
        "value": 0,
        "description": "The wavelength",
        "unit": u.um,
    },
    fov={
        "name": "fov",
        "shortname": "fov",
        "value": 0,
        "min": 0,
        "max": 100,
        "description": "The field of view",
        "unit": u.mas,
        "free": False,
    },
    fwhm={
        "name": "fwhm",
        "shortname": "fwhm",
        "value": 0,
        "min": 0,
        "max": 20,
        "description": "The full width half maximum",
        "unit": u.mas,
        "free": True,
    },
    la={
        "name": "la",
        "shortname": "la",
        "value": 0,
        "min": -1,
        "max": 1.5,
        "description": "The logarithm of the half-light/-flux radius",
        "unit": u.one,
        "free": True,
    },
    lkr={
        "name": "lkr",
        "shortname": "lkr",
        "value": 0,
        "min": -1,
        "max": 1,
        "description": "The logarithm of the kernel/ring radius",
        "unit": u.one,
        "free": True,
    },
    hlr={
        "name": "hlr",
        "shortname": "hlr",
        "value": 0,
        "min": 0,
        "max": 20,
        "description": "The half-light/-flux radius",
        "unit": u.mas,
        "free": True,
    },
    kernel={
        "name": "kernel",
        "shortname": "kernel",
        "value": 0,
        "min": 0,
        "max": 20,
        "description": "The kernel radius",
        "unit": u.mas,
        "free": True,
    },
    diam={
        "name": "diameter",
        "shortname": "diam",
        "value": 0,
        "min": 0,
        "max": 20,
        "description": "The diameter",
        "unit": u.mas,
        "free": True,
    },
    width={
        "name": "width",
        "shortname": "width",
        "value": 0,
        "min": 0,
        "max": 300,
        "description": "The width",
        "unit": u.mas,
        "free": True,
    },
    r0={
        "name": "r0",
        "shortname": "r0",
        "value": 1,
        "unit": u.au,
        "description": "Reference radius",
        "free": False,
    },
    rin={
        "name": "rin",
        "shortname": "rin",
        "value": 0,
        "unit": u.au,
        "min": 0,
        "max": 10,
        "description": "The inner radius",
        "free": True,
    },
    rout={
        "name": "rout",
        "shortname": "rout",
        "min": 0,
        "max": 30,
        "value": 4,
        "unit": u.au,
        "description": "The outer radius",
        "free": False,
    },
    c={
        "name": "c",
        "shortname": "c",
        "value": 0,
        "min": -1,
        "max": 1,
        "unit": u.one,
        "description": "An azimuthal modulation amplitude",
        "free": True,
    },
    s={
        "name": "s",
        "shortname": "s",
        "value": 0,
        "min": -1,
        "max": 1,
        "unit": u.one,
        "description": "An azimuthal modulation amplitude",
        "free": True,
    },
    temp0={
        "name": "temp0",
        "shortname": "temp0",
        "value": 0,
        "unit": u.K,
        "free": True,
        "min": 0,
        "max": 3000,
        "description": "The temperature at a reference radius",
    },
    tempc={
        "name": "tempc",
        "shortname": "tempc",
        "value": 0,
        "unit": u.K,
        "free": True,
        "min": 0,
        "max": 3000,
        "description": "The characteristic temperature",
    },
    temps={
        "name": "temps",
        "shortname": "temps",
        "value": None,
        "unit": u.K,
        "free": False,
        "min": None,
        "max": None,
        "description": "A grid of temperatures",
    },
    q={
        "name": "q",
        "shortname": "q",
        "value": 0,
        "min": 0,
        "max": 1,
        "unit": u.one,
        "free": True,
        "description": "The power-law exponent for a temperature profile",
    },
    p={
        "name": "p",
        "shortname": "p",
        "value": 0,
        "min": 0,
        "max": 1,
        "unit": u.one,
        "free": True,
        "description": "The power-law exponent for a dust surface density profile",
    },
    sigma0={
        "name": "sigma0",
        "shortname": "sigma0",
        "value": 0,
        "unit": u.g / u.cm**2,
        "free": True,
        "min": 0,
        "max": 1e-2,
        "description": "The surface density at a reference radius",
    },
    kappa_abs={
        "name": "kappa_abs",
        "shortname": "kappa_abs",
        "value": 0,
        "unit": u.cm**2 / u.g,
        "free": False,
        "smooth": True,
        "description": "The dust mass absorption coefficient",
    },
    kappa_cont={
        "name": "kappa_cont",
        "shortname": "kappa_cont",
        "value": 0,
        "unit": u.cm**2 / u.g,
        "free": False,
        "smooth": True,
        "description": "The continuum dust mass absorption coefficient",
    },
    cont_weight={
        "name": "cont_weight",
        "shortname": "cont_weight",
        "value": 0,
        "unit": u.pct,
        "free": True,
        "min": 0,
        "max": 100,
        "description": "The mass fraction of continuum vs. silicate",
    },
    pah={
        "name": "pah",
        "shortname": "pah",
        "value": 0,
        "unit": u.Jy,
        "free": False,
        "description": "The flux curve of the PAHs",
    },
    dist={
        "name": "dist",
        "shortname": "dist",
        "value": 0,
        "unit": u.pc,
        "free": False,
        "min": 0,
        "max": 1000,
        "description": "The Distance",
    },
    eff_temp={
        "name": "eff_temp",
        "shortname": "eff_temp",
        "value": 0,
        "min": 0,
        "max": 30000,
        "unit": u.K,
        "free": False,
        "description": "The star's effective temperature",
    },
    eff_radius={
        "name": "eff_radius",
        "shortname": "eff_radius",
        "value": 0,
        "min": 0,
        "max": 10,
        "unit": u.Rsun,
        "free": False,
        "description": "The stellar radius",
    },
)


# NOTE: Data
vis = SimpleNamespace(
    value=np.array([]),
    err=np.array([]),
    ucoord=np.array([]).reshape(1, -1),
    vcoord=np.array([]).reshape(1, -1),
)
vis2 = SimpleNamespace(
    value=np.array([]),
    err=np.array([]),
    ucoord=np.array([]).reshape(1, -1),
    vcoord=np.array([]).reshape(1, -1),
)
t3 = SimpleNamespace(
    value=np.array([]), err=np.array([]), u123coord=np.array([]), v123coord=np.array([])
)
flux = SimpleNamespace(value=np.array([]), err=np.array([]))
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
    flux=flux,
    vis=vis,
    vis2=vis2,
    t3=t3,
    gravity=gravity,
    binning=binning,
    dtype=dtype,
    interpolation=interpolation,
)

# NOTE: Model
model = SimpleNamespace(
    components_and_params=None,
    constant_params={},
    shared_params=None,
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

# NOTE: Fitting
weights = SimpleNamespace(flux=1, t3=1, vis=1)
fit = SimpleNamespace(
    weights=weights,
    data=["flux", "vis", "t3"],
    method="dynesty",
    wavelengths=None,
    quantiles=[2.5, 50, 97.5],
)

# NOTE: All options
OPTIONS = SimpleNamespace(data=data, model=model, plot=plot, fit=fit)
