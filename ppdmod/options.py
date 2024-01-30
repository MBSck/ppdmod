from typing import List, Optional
from types import SimpleNamespace

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib import colormaps as mcm


def convert_style_to_colormap(style: str) -> ListedColormap:
    """Converts a style into a colormap."""
    plt.style.use(style)
    colormap = ListedColormap(
            plt.rcParams["axes.prop_cycle"].by_key()["color"])
    plt.style.use("default")
    return colormap


def get_colormap(colormap: str) -> ListedColormap:
    """Gets the colormap as the matplotlib colormaps or styles."""
    try:
        return mcm.get_cmap(colormap)
    except ValueError:
        return convert_style_to_colormap(colormap)


def get_colorlist(colormap: str, ncolors: Optional[int] = 10) -> List[str]:
    """Gets the colormap as a list from the matplotlib colormaps."""
    return [get_colormap(colormap)(i) for i in range(ncolors)]


# NOTE: A list of standard parameters to be used when defining new components.
# TODO: Make this into a simple namespace as well
STANDARD_PARAMETERS = {
        "x": {"name": "x", "shortname": "x",
              "value": 0, "description": "x position",
              "unit": u.mas, "free": False},
        "y": {"name": "y", "shortname": "y",
              "value": 0, "description": "y position",
              "unit": u.mas, "free": False},
        "f": {"name": "flux", "shortname": "f",
              "value": None, "description": "The wavelength dependent flux",
              "unit": u.Jy, "free": False},
        "elong": {"name": "elong", "shortname": "elong",
                  "value": 1, "description": "Inclination of the object", "unit": u.one},
        "pa": {"name": "pa", "shortname": "pa",
               "value": 0, "description": "Major-axis position angle", "unit": u.deg},
        "pixel_size": {"name": "pixel_size", "shortname": "pixsize",
                       "value": 1, "description": "Pixel size", "unit": u.mas, "free": False},
        "dim": {"name": "dim", "shortname": "dim",
                "value": 128, "description": "Dimension",
                "unit": u.one, "dtype": int, "free": False},
        "wl": {"name": "wl", "shortname": "wl",
               "value": 0, "description": "Wavelength", "unit": u.m},
        "fov": {"name": "fov", "shortname": "fov",
                "value": 0, "description": "The interferometric field of view",
                "unit": u.mas, "free": False},
        "rin0": {"name": "rin0", "shortname": "rin0",
                 "value": 0, "unit": u.mas,
                 "description": "Inner radius of the whole disk", "free": False},
        "rin": {"name": "rin", "shortname": "rin",
                "value": 0, "unit": u.mas,
                "description": "Innermost radius of the component", "free": True},
        "rout": {"name": "rout", "shortname": "rout",
                 "value": 300, "unit": u.mas,
                 "description": "Outer radius of the component", "free": False},
        "a": {"name": "a", "shortname": "a",
              "value": 0, "unit": u.one,
              "description": "Azimuthal modulation amplitude", "free": True},
        "phi": {"name": "phi", "shortname": "phi",
                "value": 0, "unit": u.deg,
                "description": "Azimuthal modulation angle", "free": True},
        "inner_temp": {"name": "inner_temp", "shortname": "rimtemp",
                       "value": 0, "unit": u.K, "free": True,
                       "description": "Inner temperature of the whole disk"},
        "q": {"name": "q", "shortname": "q",
              "value": 0, "unit": u.one, "free": True,
              "description": "Power-law exponent for the temperature profile"},
        "p": {"name": "p", "shortname": "p",
              "value": 0, "unit": u.one, "free": True,
              "description": "Power-law exponent for the dust surface density profile"},
        "inner_sigma": {"name": "inner_sigma", "shortname": "rimsigma",
                        "value": 0, "unit": u.g/u.cm**2, "free": True,
                        "description": "Inner surface density"},
        "kappa_abs": {"name": "kappa_abs", "shortname": "kappaabs",
                      "value": 0, "unit": u.cm**2/u.g, "free": False,
                      "description": "Dust mass absorption coefficient"},
        "kappa_cont": {"name": "kappa_cont", "shortname": "kappacon",
                       "value": 0, "unit": u.cm**2/u.g, "free": False,
                       "description": "Continuum dust mass absorption coefficient"},
        "cont_weight": {"name": "cont_weight", "shortname": "conwei",
                        "value": 0, "unit": u.one, "free": True,
                        "description": "Mass fraction of continuum vs. silicate"},
        "dist": {"name": "dist", "shortname": "dist",
                 "value": 0, "unit": u.pc, "free": False,
                 "description": "Distance to the star"},
        "eff_temp": {"name": "eff_temp", "shortname": "efftemp",
                     "value": 0, "unit": u.K, "free": False,
                     "description": "The star's effective temperature"},
        "eff_radius": {"name": "eff_radius", "shortname": "effrad",
                       "value": 0, "unit": u.Rsun, "free": False,
                       "description": "The stellar radius"},
}


# NOTE: Data
vis = SimpleNamespace(value=np.array([]), err=np.array([]),
                      ucoord=np.array([]), vcoord=np.array([]))
vis2 = SimpleNamespace(value=np.array([]), err=np.array([]),
                       ucoord=np.array([]), vcoord=np.array([]))
t3 = SimpleNamespace(value=np.array([]), err=np.array([]),
                     u123coord=np.array([]), v123coord=np.array([]))
flux = SimpleNamespace(value=np.array([]), err=np.array([]))
binning = SimpleNamespace(window=0.1)
gravity = SimpleNamespace(index=20)
dtype = SimpleNamespace(complex=np.complex64, real=np.float32)
data = SimpleNamespace(readouts=[], flux=flux, vis=vis,
                       vis2=vis2, t3=t3, gravity=gravity,
                       binning=binning, dtype=dtype)

# NOTE: Model
model = SimpleNamespace(components_and_params={},
                        constant_params={}, shared_params={},
                        dtype=dtype, gridtype="linear", modulation=0)


# NOTE: Spectrum
coefficients = SimpleNamespace(
        low=[0.10600484,  0.01502548,  0.00294806, -0.00021434],
        high=[-8.02282965e-05,  3.83260266e-03, 7.60090459e-05, -4.30753848e-07])
kernel = SimpleNamespace(width=10)
spectrum = SimpleNamespace(binning=7, coefficients=coefficients,
                           kernel=kernel)

# NOTE: Plot
color = SimpleNamespace(background="white",
                        colormap="tab20", number=100,
                        list=get_colorlist("tab20", 100))
errorbar = SimpleNamespace(color="",
                           markeredgecolor="black",
                           markeredgewidth=0.2,
                           capsize=5, capthick=3,
                           ecolor="gray", zorder=2)
scatter = SimpleNamespace(color="", edgecolor="black",
                          linewidths=0.2, zorder=3)
plot = SimpleNamespace(dpi=300, color=color,
                       errorbar=errorbar, scatter=scatter)

# NOTE: Fitting
weights = SimpleNamespace(cphase=1, flux=1, t3=1, vis=1)
fit = SimpleNamespace(weights=weights,
                      data=["flux", "vis", "t3"],
                      method="emcee",
                      wavelengths=[],
                      quantiles=[16, 50, 84])

# NOTE: All options
OPTIONS = SimpleNamespace(data=data, model=model,
                          spectrum=spectrum,
                          plot=plot, fit=fit)
