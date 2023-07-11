from dataclasses import dataclass
from typing import Optional, List

import astropy.units as u
import numpy as np
from astropy.units import Quantity


# NOTE: Here is a list of standard parameters to be used when defining new components
_standardParameters = {
    "x": {"name": "x", "value": 0, "description": "x position", "unit": u.mas, "free": False},
    "y": {"name": "y", "value": 0, "description": "y position", "unit": u.mas, "free": False},
    "f": {"name": "f", "value": 1, "description": "flux", "unit": u.one},
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


@dataclass
class Parameter:
    name: str
    value: any
    description: str
    unit: Quantity
    free: bool = True
    limits: List[int] = None
    wavelengths: np.ndarray = None

    def __call__(self,
                 wavelengths: Optional[np.ndarray] = None) -> np.ndarray:
        """Gets the value for the parameter or the corresponding
        values for the wavelengths."""
        if self.wavelengths is None:
            return self.value
        return self.value[np.where(self.wavelengths == wavelengths)]

    def __str__(self):
        return f"Parameter: {self.name} has the value {self.value} and "\
               f"is {'free' if self.free else 'not free'} "\
               f"with its limits being {self.range_min}-{self.range_max}."

    def __repr__(self):
        return self.__str__()


if __name__ == "__main__":
    x = Parameter(**_standardParameters["x"])
    breakpoint()
