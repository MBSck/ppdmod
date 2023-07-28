from dataclasses import dataclass
from typing import Optional

import astropy.units as u
import numpy as np
from numpy.typing import ArrayLike


# NOTE: Here is a list of standard parameters to be used when defining new components
STANDARD_PARAMETERS = {
    "x": {"name": "x", "value": 0, "description": "x position", "unit": u.mas, "free": False},
    "y": {"name": "y", "value": 0, "description": "y position", "unit": u.mas, "free": False},
    "r": {"name": "r", "value": 0, "description": "radius", "unit": u.mas},
    "elong": {"name": "elong", "value": 1, "description": "Elongation Ratio", "unit": u.one},
    "pa": {"name": "pa", "value": 0, "description": "Major-axis Position angle", "unit": u.deg},
    "pixSize": {"name": "pixSize", "value": 0, "description": "Pixel Size", "unit": u.mas, "free": False},
    "dim": {"name": "dim", "value": 128, "description": "Dimension in pixels", "unit": u.one, "free": False},
    "wl": {"name": "wl", "value": 0, "description": "Wavelength", "unit": u.m},
    "scale": {"name": "scale", "value": 1, "description": "Scaling Factor", "unit": u.one},
    "index": {"name": "index", "value": 1, "description": "Index", "unit": u.one},
    "fov": {"name": "fov", "value": 0, "description": "The interferometric field of view", "unit": u.mas, "free": False},
    "amp": {"name": "amplitude", "value": 1, "description": "Amplitude", "unit": u.one},
    "exp": {"name": "exp", "value": 0, "description": "Power-law Exponent", "unit": u.one},
}


@dataclass(kw_only=True)
class Parameter:
    """Defines a parameter."""
    name: str
    value: any
    description: str
    unit: u.Quantity
    free: bool = True
    min: float = None
    max: float = None
    wavelength: np.ndarray = None

    def __post__init__(self):
        """Post initialisation actions."""
        self.value = self._set_to_numpy_array(self.wavelength)
        self.wavelength = self._set_to_numpy_array(self.wavelength)

    def __call__(self,
                 wavelength: Optional[np.ndarray] = None) -> np.ndarray:
        """Gets the value for the parameter or the corresponding
        values for the wavelengths."""
        if self.wavelength is None:
            return self.value[0]
        return self.value[np.where(self.wavelength == wavelength)]

    def __str__(self):
        message = f"Parameter: {self.name} has the value {self.value} and "\
                  f"is {'free' if self.free else 'not free'}"
        if self.max is not None:
            message += f" with its limits being {self.min}-{self.max}"
        return message

    def _set_to_numpy_array(self, array: ArrayLike) -> np.ndarray:
        """Converts a value to a numpy array."""
        if not isinstance(array, np.ndarray):
            if isinstance(array, (tuple, list)):
                array = np.array(array)
            else:
                array = np.array([array])
        return array

    def set(self, min: Optional[float] = None,
            max: Optional[float] = None) -> None:
        """Sets the limits of the parameters."""
        self.min, self.max = min, max
