from dataclasses import dataclass
from typing import Union, Optional, Any

import astropy.units as u
import numpy as np
from numpy.typing import ArrayLike

from .utils import get_closest_indices


# NOTE: A list of standard parameters to be used when defining new components.
STANDARD_PARAMETERS = {
    "x": {"name": "x", "value": 0, "description": "x position",
          "unit": u.mas, "free": False},
    "y": {"name": "y", "value": 0, "description": "y position",
          "unit": u.mas, "free": False},
    "elong": {"name": "elong", "value": 1,
              "description": "Elongation Ratio", "unit": u.one},
    "pa": {"name": "pa", "value": 0,
           "description": "Major-axis Position angle", "unit": u.deg},
    "pixel_size": {"name": "pixel_size", "value": 1,
                   "description": "Pixel Size", "unit": u.mas, "free": False},
    "dim": {"name": "dim", "value": 128,
            "description": "Dimension in pixels",
            "unit": u.one, "dtype": int, "free": False},
    "wl": {"name": "wl", "value": 0,
           "description": "Wavelength", "unit": u.m},
    "fov": {"name": "fov", "value": 0,
            "description": "The interferometric field of view",
            "unit": u.mas, "free": False},
    "rin": {"name": "rin", "value": 0, "unit": u.mas,
            "description": "Inner radius of the disk", "free": True},
    "rout": {"name": "rout", "value": np.inf, "unit": u.mas,
             "description": "Inner radius of the disk", "free": False},
    "a": {"name": "a", "value": 0, "unit": u.one,
          "description": "Azimuthal modulation amplitude", "free": True},
    "phi": {"name": "a", "value": 0, "unit": u.deg,
            "description": "Azimuthal modulation angle", "free": True},
    "inner_temp": {"name": "inner_temp", "value": 0,
                   "unit": u.K, "free": True,
                   "description": "Inner temperature"},
    "q": {"name": "q", "value": 0, "unit": u.one, "free": True,
          "description": "Power-law exponent for the temperature profile"},
    "p": {"name": "p", "value": 0, "unit": u.one, "free": True,
          "description": "Power-law exponent for the dust surface density profile"},
    "inner_sigma": {"name": "inner_sigma", "value": 0,
                    "unit": u.g/u.cm**2, "free": True,
                    "description": "Inner surface density"},
    "kappa_abs": {"name": "kappa_abs", "value": 0,
                  "unit": u.cm**2/u.g, "free": False,
                  "description": "Dust mass absorption coefficient"},
    "kappa_cont": {"name": "kappa_cont", "value": 0,
                   "unit": u.cm**2/u.g, "free": False,
                   "description": "Continuum dust mass absorption coefficient"},
    "cont_weight": {"name": "cont_weight", "value": 0,
                    "unit": u.one, "free": True,
                    "description": "Dust mass continuum absorption coefficient's weight"},
    "dist": {"name": "dist", "value": 0,
             "unit": u.pc, "free": False,
             "description": "Distance of the star"},
    "eff_temp": {"name": "eff_temp", "value": 0,
                 "unit": u.K, "free": False,
                 "description": "The star's effective Temperature"},
    "eff_radius": {"name": "eff_radius", "value": 0,
                   "unit": u.Rsun, "free": False,
                   "description": "The stellar radius"},
}


@dataclass(kw_only=True)
class Parameter:
    """Defines a parameter."""
    name: str
    value: Any
    unit: u.Quantity
    free: Optional[bool] = True
    description: Optional[str] = None
    min: Optional[float] = None
    max: Optional[float] = None
    dtype: Optional[type] = None
    wavelength: Optional[u.um] = None

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

    def __call__(self, wavelength: Optional[u.um] = None) -> np.ndarray:
        """Gets the value for the parameter or the corresponding
        values for the wavelengths."""
        if self.wavelength is None:
            value = self.value
        else:
            # Hack: Multiplying by microns makes it work.
            indices = list(get_closest_indices(
                wavelength, array=self.wavelength*u.um).values())
            value = self.value[indices]
            value = value[0] if len(value) == 1 else value
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
