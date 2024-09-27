from dataclasses import dataclass
from typing import Union, Optional, Any

import astropy.units as u
import numpy as np
from numpy.typing import ArrayLike

from .options import OPTIONS
from .utils import smooth_interpolation, get_band


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
    grid: Optional[np.ndarray] = None
    smooth: Optional[bool] = False

    def __setattr__(self, key: str, value: Any):
        """Sets an attribute."""
        if key != "unit":
            if isinstance(value, u.Quantity):
                value = value.value
        super().__setattr__(key, value)

    def __post_init__(self):
        """Post initialisation actions."""
        self.value = self._set_to_numpy_array(self.value)
        self.grid = self._set_to_numpy_array(self.grid)

    def __call__(self, points: Optional[u.Quantity] = None) -> np.ndarray:
        """Gets the value for the parameter or the corresponding
        values for some points."""
        if points is None or self.grid is None:
            value = self.value
        else:
            # HACK: Split the bands as the convolution is only done for LM and N band
            # Temps are not in this scheme and should also not be convolved (they are
            # smoothly interpolated)
            if OPTIONS.model.convolve and self.name != "temps":
                bands = np.array(list(map(get_band, points)))
                if "hband" in bands and "kband" in bands:
                    condition = (bands == "hband") | (bands == "kband")
                if "hband" in bands:
                    condition = bands == "hband"
                elif "kband" in bands:
                    condition = bands == "kband"
                else:
                    condition = None

                value = None
                if condition is not None:
                    points_hkband = points[np.where(condition)]

                    if self.smooth:
                        value = smooth_interpolation(
                            points_hkband.value, self.grid, self.value)
                    else:
                        value = np.interp(
                            points_hkband.value, self.grid, self.value)

                if condition is not None:
                    points_other = points[np.where(~condition)]
                else:
                    points_other = points

                convolved_value = np.interp(
                    points_other.value, self.grid, self.value)

                if value is not None:
                    value = np.vstack((value, convolved_value))
                else:
                    value = convolved_value
            else:
                if self.smooth:
                    value = smooth_interpolation(points.value, self.grid, self.value)
                else:
                    value = np.interp(points.value, self.grid, self.value)

        return u.Quantity(value, unit=self.unit, dtype=self.dtype)

    def __str__(self):
        message = f"Parameter: {self.name} has the value "\
            f"{np.round(self.value, 2)} and "\
            f"is {'free' if self.free else 'fixed'}"
        if self.max is not None:
            message += f" with its limits being {self.min:.1f}-{self.max:.1f}"
        return message

    def _set_to_numpy_array(
        self, array: Optional[ArrayLike] = None,
        retain_value: Optional[bool] = False) -> Union[Any, np.ndarray]:
        """Converts a value to a numpy array."""
        if array is None:
            return

        if isinstance(array, u.Quantity) and retain_value:
            return array

        if not isinstance(array, np.ndarray):
            if isinstance(array, (tuple, list)):
                return np.array(array)

        return array

    # TODO: One can make this modular, maybe cool for oimodeler?
    def set(self, min: Optional[float] = None,
            max: Optional[float] = None) -> None:
        """Sets the limits of the parameters."""
        self.min, self.max = min, max
