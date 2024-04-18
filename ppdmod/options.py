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
STANDARD_PARAMETERS = SimpleNamespace(
        x={"name": "x", "shortname": "x",
           "value": 0, "description": "The x position",
           "min": -10, "max": 10,
           "unit": u.mas, "free": False},
        y={"name": "y", "shortname": "y",
           "value": 0, "description": "The y position",
           "min": -10, "max": 10,
           "unit": u.mas, "free": False},
        f={"name": "flux", "shortname": "f",
           "value": None, "description": "The flux",
           "min": 0, "max": 10,
           "unit": u.Jy, "free": False},
        fr={"name": "flux ratio", "shortname": "fr",
            "value": 1, "description": "The flux ratio",
            "min": 0, "max": 1, "unit": u.one, "free": False},
        inc={"name": "inclination", "shortname": "inc", "value": 1,
             "min": 0, "max": 1, "unit": u.one,
             "description": "The cosine of the inclination"},
        pa={"name": "pa", "shortname": "pa", "value": 0,
            "min": 0, "max": 180, "unit": u.deg,
            "description": "The major-axis position angle"},
        pixel_size={"name": "pixel_size", "shortname": "pixsize",
                    "value": 1, "min": 0, "max": 100,
                    "description": "The pixel size", "unit": u.mas, "free": False},
        dim={"name": "dim", "shortname": "dim",
             "value": 128, "description": "The pixel dimension",
             "unit": u.one, "dtype": int, "free": False},
        exp={"name": "exponent", "shortname": "exp",
             "value": 1, "min": 0, "max": 1, "unit": u.one,
             "description": "An exponent", "free": True},
        wl={"name": "wl", "shortname": "wl",
            "value": 0, "description": "The wavelength", "unit": u.um},
        fov={"name": "fov", "shortname": "fov",
             "value": 0, "min": 0, "max": 100,
             "description": "The field of view",
             "unit": u.mas, "free": False},
        fwhm={"name": "fwhm", "shortname": "fwhm",
              "value": 0, "min": 0, "max": 20,
              "description": "The full width half maximum",
              "unit": u.mas, "free": True},
        diam={"name": "diameter", "shortname": "diam",
              "value": 0, "min": 0, "max": 20,
              "description": "The diameter",
              "unit": u.mas, "free": True},
        width={"name": "width", "shortname": "width",
               "value": 0, "min": 0, "max": 300,
               "description": "The width",
               "unit": u.mas, "free": True},
        r0={"name": "r0", "shortname": "r0",
            "value": 0, "unit": u.mas,
            "description": "Reference radius", "free": True},
        rin={"name": "rin", "shortname": "rin",
             "value": 0, "unit": u.mas,
             "min": 0, "max": 30,
             "description": "The inner radius", "free": True},
        rout={"name": "rout", "shortname": "rout",
              "min": 0, "max": 300,
              "value": 300, "unit": u.mas,
              "description": "The outer radius", "free": False},
        a={"name": "a", "shortname": "a", "value": 0,
           "min": 0, "max": 1, "unit": u.one,
           "description": "The azimuthal modulation amplitude", "free": True},
        phi={"name": "phi", "shortname": "phi",
             "value": 0, "min": -180, "max": 180, "unit": u.deg,
             "description": "The azimuthal modulation angle", "free": True},
        inner_temp={"name": "inner_temp", "shortname": "rimtemp",
                    "value": 0, "unit": u.K, "free": True,
                    "description": "The inner temperature"},
        q={"name": "q", "shortname": "q", "value": 0,
           "min": 0, "max": 1, "unit": u.one, "free": True,
           "description": "The power-law exponent for a temperature profile"},
        p={"name": "p", "shortname": "p", "value": 0,
           "min": 0, "max": 1, "unit": u.one, "free": True,
           "description": "The power-law exponent for a dust surface density profile"},
        inner_sigma={"name": "inner_sigma", "shortname": "rimsigma",
                     "value": 0, "unit": u.g/u.cm**2, "free": True,
                     "description": "The inner surface density"},
        kappa_abs={"name": "kappa_abs", "shortname": "kappaabs",
                   "value": 0, "unit": u.cm**2/u.g, "free": False,
                   "description": "The dust mass absorption coefficient"},
        kappa_cont={"name": "kappa_cont", "shortname": "kappacon",
                    "value": 0, "unit": u.cm**2/u.g, "free": False,
                    "description": "The continuum dust mass absorption coefficient"},
        cont_weight={"name": "cont_weight", "shortname": "conwei",
                     "value": 0, "unit": u.one, "free": True,
                     "description": "The mass fraction of continuum vs. silicate"},
        dist={"name": "dist", "shortname": "dist",
              "value": 0, "unit": u.pc, "free": False,
              "min": 0, "max": 1000, "description": "The Distance"},
        eff_temp={"name": "eff_temp", "shortname": "efftemp",
                  "value": 0, "min": 0, "max": 30000, "unit": u.K, "free": False,
                  "description": "The star's effective temperature"},
        eff_radius={"name": "eff_radius", "shortname": "effrad",
                    "value": 0, "min": 0, "max": 10,
                    "unit": u.Rsun, "free": False,
                    "description": "The stellar radius"},
)


# NOTE: Data
vis = SimpleNamespace(value=np.array([]), err=np.array([]),
                      ucoord=np.array([]).reshape(1, -1),
                      vcoord=np.array([]).reshape(1, -1))
vis2 = SimpleNamespace(value=np.array([]), err=np.array([]),
                       ucoord=np.array([]).reshape(1, -1),
                       vcoord=np.array([]).reshape(1, -1))
t3 = SimpleNamespace(value=np.array([]), err=np.array([]),
                     u123coord=np.array([]), v123coord=np.array([]))
flux = SimpleNamespace(value=np.array([]), err=np.array([]))
gravity = SimpleNamespace(index=20)
dtype = SimpleNamespace(complex=np.complex64, real=np.float32)
data = SimpleNamespace(readouts=[], flux=flux, vis=vis,
                       vis2=vis2, t3=t3, gravity=gravity,
                       binning=0.1*u.um, dtype=dtype)

# NOTE: Model
model = SimpleNamespace(components_and_params=None,
                        constant_params=None, shared_params=None,
                        output="physical", reference_radius=1*u.au,
                        gridtype="linear", modulation=0)

# NOTE: Plot
color = SimpleNamespace(background="white",
                        colormap="tab20", number=100,
                        list=get_colorlist("tab20", 100))
errorbar = SimpleNamespace(color=None,
                           markeredgecolor="black",
                           markeredgewidth=0.2,
                           capsize=5, capthick=3,
                           ecolor="gray", zorder=2)
scatter = SimpleNamespace(color="", edgecolor="black",
                          linewidths=0.2, zorder=3)
plot = SimpleNamespace(dpi=300, color=color,
                       errorbar=errorbar, scatter=scatter)

# NOTE: Fitting
weights = SimpleNamespace(flux=1, t3=1, vis=1)
fit = SimpleNamespace(weights=weights,
                      data=["flux", "vis", "t3"],
                      method="emcee",
                      wavelengths=None,
                      quantiles=[16, 50, 84])

# NOTE: All options
OPTIONS = SimpleNamespace(data=data, model=model,
                          plot=plot, fit=fit)
