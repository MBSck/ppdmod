from dataclasses import dataclass
from typing import Union, Optional, Any

import astropy.units as u
import numpy as np
from numpy.typing import ArrayLike


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
    "pixel_size": {"name": "pixel_size", "value": 0,
                   "description": "Pixel Size", "unit": u.mas, "free": False},
    "dim": {"name": "dim", "value": 128,
            "description": "Dimension in pixels",
            "unit": u.Quantity(value=1, unit=u.one, dtype=int), "free": False},
    "wl": {"name": "wl", "value": 0,
           "description": "Wavelength", "unit": u.m},
    "fov": {"name": "fov", "value": 0,
            "description": "The interferometric field of view",
            "unit": u.mas, "free": False},
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
    wavelength: Optional[np.ndarray] = None

    def __setattr__(self, key: str, value: Any):
        """Sets an attribute."""
        if key != "unit":
            if isinstance(value, u.Quantity):
                value = value.value
        super().__setattr__(key, value)

    def __post_init__(self):
        """Post initialisation actions."""
        self.value = self._set_to_numpy_array(self.value)
        self.wavelength = self._set_to_numpy_array(self.wavelength)

    def __call__(self,
                 wavelength: Optional[Union[float, np.ndarray]] = None
                 ) -> np.ndarray:
        """Gets the value for the parameter or the corresponding
        values for the wavelengths."""
        if wavelength is not None:
            wavelength = wavelength.value\
                if isinstance(wavelength, u.Quantity) else wavelength
        if self.wavelength is None:
            value = self.value
        else:
            value = self.value[np.where(self.wavelength == wavelength)]
            if value.size == 0:
                index = np.abs(self.wavelength - wavelength).argmin()
                value = self.value[index]
            value = value[0] if len(value) == 1 else value
        return value*self.unit

    def __str__(self):
        message = f"Parameter: {self.name} has the value {self.value} and "\
                  f"is {'free' if self.free else 'not free'}"
        if self.max is not None:
            message += f" with its limits being {self.min}-{self.max}"
        return message

    def _set_to_numpy_array(self,
                            array: Optional[ArrayLike] = None
                            ) -> Union[Any, np.ndarray]:
        """Converts a value to a numpy array."""
        if not isinstance(array, np.ndarray):
            if isinstance(array, (tuple, list)):
                return np.array(array)
        return array

    def set(self, min: Optional[float] = None,
            max: Optional[float] = None) -> None:
        """Sets the limits of the parameters."""
        self.min, self.max = min, max


if __name__ == "__main__":
    x = Parameter(name="x", description="", value=0*u.mas, unit=u.mas)
    breakpoint()
