from dataclasses import dataclass
from typing import Union, Optional, Any

import astropy.units as u
import numpy as np
from numpy.typing import ArrayLike

from .utils import get_closest_indices
from .options import OPTIONS


# NOTE: A list of standard parameters to be used when defining new components.
STANDARD_PARAMETERS = {
        "x": {"name": "x", "shortname": "x",
              "value": 0, "description": "x position (mas).",
              "unit": u.mas, "free": False},
        "y": {"name": "y", "shortname": "y",
              "value": 0, "description": "y position (mas).",
              "unit": u.mas, "free": False},
        "f": {"name": "flux", "shortname": "f",
              "value": None, "description": "The wavelength dependent flux (Jy).",
              "unit": u.Jy, "free": False},
        "elong": {"name": "elong", "shortname": "elong",
                  "value": 1, "description": "Inclination of the object.", "unit": u.one},
        "pa": {"name": "pa", "shortname": "pa",
               "value": 0, "description": "Major-axis position angle (deg).", "unit": u.deg},
        "pixel_size": {"name": "pixel_size", "shortname": "pixsize",
                       "value": 1, "description": "Pixel size (mas).", "unit": u.mas, "free": False},
        "dim": {"name": "dim", "shortname": "dim",
                "value": 128, "description": "Dimension (px).",
                "unit": u.one, "dtype": int, "free": False},
        "wl": {"name": "wl", "shortname": "wl",
               "value": 0, "description": "Wavelength (m).", "unit": u.m},
        "fov": {"name": "fov", "shortname": "fov",
                "value": 0, "description": "The interferometric field of view (mas).",
                "unit": u.mas, "free": False},
        "rin0": {"name": "rin0", "shortname": "rin0",
                 "value": 0, "unit": u.mas,
                 "description": "Inner radius of the whole disk (mas).", "free": False},
        "rin": {"name": "rin", "shortname": "rin",
                "value": 0, "unit": u.mas,
                "description": "Innermost radius of the component (mas).", "free": True},
        "rout": {"name": "rout", "shortname": "rout",
                 "value": 300, "unit": u.mas,
                 "description": "Outer radius of the component (mas).", "free": False},
        "a": {"name": "a", "shortname": "a",
              "value": 0, "unit": u.one,
              "description": "Azimuthal modulation amplitude.", "free": True},
        "phi": {"name": "phi", "shortname": "phi",
                "value": 0, "unit": u.deg,
                "description": "Azimuthal modulation angle (deg).", "free": True},
        "inner_temp": {"name": "inner_temp", "shortname": "rimtemp",
                       "value": 0, "unit": u.K, "free": True,
                       "description": "Inner temperature of the whole disk (K)."},
        "q": {"name": "q", "shortname": "q",
              "value": 0, "unit": u.one, "free": True,
              "description": "Power-law exponent for the temperature profile."},
        "p": {"name": "p", "shortname": "p",
              "value": 0, "unit": u.one, "free": True,
              "description": "Power-law exponent for the dust surface density profile."},
        "inner_sigma": {"name": "inner_sigma", "shortname": "rimsigma",
                        "value": 0, "unit": u.g/u.cm**2, "free": True,
                        "description": "Inner surface density (g/cm^2)."},
        "kappa_abs": {"name": "kappa_abs", "shortname": "kappaabs",
                      "value": 0, "unit": u.cm**2/u.g, "free": False,
                      "description": "Dust mass absorption coefficient (cm^2/g)."},
        "kappa_cont": {"name": "kappa_cont", "shortname": "kappacon",
                       "value": 0, "unit": u.cm**2/u.g, "free": False,
                       "description": "Continuum dust mass absorption coefficient (cm^2/g)."},
        "cont_weight": {"name": "cont_weight", "shortname": "conwei",
                        "value": 0, "unit": u.one, "free": True,
                        "description": "Dust mass continuum absorption coefficient's weight."},
    "dist": {"name": "dist", "shortname": "dist",
             "value": 0, "unit": u.pc, "free": False,
             "description": "Distance to the star (pc)."},
    "eff_temp": {"name": "eff_temp", "shortname": "efftemp",
                 "value": 0, "unit": u.K, "free": False,
                 "description": "The star's effective temperature (K)."},
    "eff_radius": {"name": "eff_radius", "shortname": "effrad",
                   "value": 0, "unit": u.Rsun, "free": False,
                   "description": "The stellar radius"},
}


@dataclass()
class Parameter:
    """Defines a parameter."""
    name: str
    value: Any
    unit: u.Quantity
    shortname: Optional[str] = None
    free: Optional[bool] = True
    description: Optional[str] = None
    min: Optional[float] = None
    max: Optional[float] = None
    dtype: Optional[type] = None
    wavelength: Optional[u.Quantity[u.um]] = None

    def __setattr__(self, key: str, value: Any):
        """Sets an attribute."""
        if key != "unit":
            if isinstance(value, u.Quantity):
                value = value.value
        super().__setattr__(key, value)

    def __post_init__(self):
        """Post initialisation actions."""
        self.value = self._set_to_numpy_array(self.value)
        self.wavelength = self._set_to_numpy_array(self.wavelength, True)

    def __call__(self, wavelength: Optional[u.Quantity[u.um]] = None) -> np.ndarray:
        """Gets the value for the parameter or the corresponding
        values for the wavelengths."""
        if wavelength is None or self.wavelength is None:
            value = self.value
        else:
            # Hack: Multiplying by microns makes it work.
            indices = list(get_closest_indices(
                wavelength, array=self.wavelength*u.um,
                window=OPTIONS["data.binning.window"]).values())
            value = self.value[indices[0]].mean()
        return u.Quantity(value, unit=self.unit, dtype=self.dtype)

    def __str__(self):
        message = f"Parameter: {self.name} has the value {self.value} and "\
                  f"is {'free' if self.free else 'not free'}"
        if self.max is not None:
            message += f" with its limits being {self.min}-{self.max}"
        return message

    def _set_to_numpy_array(self,
                            array: Optional[ArrayLike] = None,
                            retain_value: Optional[bool] = False
                            ) -> Union[Any, np.ndarray]:
        """Converts a value to a numpy array."""
        if array is None:
            return
        if isinstance(array, u.Quantity) and retain_value:
            return array
        if not isinstance(array, np.ndarray):
            if isinstance(array, (tuple, list)):
                return np.array(array)
        return array

    def set(self, min: Optional[float] = None,
            max: Optional[float] = None) -> None:
        """Sets the limits of the parameters."""
        self.min, self.max = min, max
