from dataclasses import dataclass
from typing import Any, Union

import astropy.units as u
import numpy as np
from numpy.typing import ArrayLike

from .options import STANDARD_PARAMS
from .utils import smooth_interpolation


@dataclass()
class Parameter:
    """Defines a parameter."""

    name: str | None = None
    shortname: str | None = None
    description: str | None = None
    value: Any | None = None
    grid: np.ndarray | None = None
    unit: u.Quantity | None = None
    min: float | None = None
    max: float | None = None
    dtype: type | None = None
    smooth: bool | None = None
    free: bool | None = None
    shared: bool | None = None
    base: str | None = None

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
        self._process_base(self.base)

    def _process_base(self, base: str | None) -> None:
        """Process the template attribute."""
        if base is None:
            return

        base_param = getattr(STANDARD_PARAMS, base)
        for key, value in base_param.items():
            if getattr(self, key) is None:
                setattr(self, key, value)

        for key in ["free", "shared", "smooth"]:
            if key not in base_param:
                if getattr(self, key) is not None:
                    continue

                setattr(self, key, False)

    def __call__(self, points: u.Quantity | None = None) -> np.ndarray:
        """Gets the value for the parameter or the corresponding
        values for some points."""
        if points is None or self.grid is None:
            value = self.value
        else:
            if self.smooth:
                value = smooth_interpolation(points.value, self.grid, self.value)
            else:
                value = np.interp(points.value, self.grid, self.value)

        return u.Quantity(value, unit=self.unit, dtype=self.dtype)

    def __str__(self):
        message = (
            f"Parameter: {self.name} has the value "
            f"{np.round(self.value, 2)} and "
            f"is {'free' if self.free else 'fixed'}"
        )
        if self.max is not None:
            message += f" with its limits being {self.min:.1f}-{self.max:.1f}"
        return message

    def _set_to_numpy_array(
        self, array: ArrayLike | None = None, retain_value: bool = False
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

    def copy(self) -> "Parameter":
        """Copies the parameter."""
        return Parameter(
            name=self.name,
            shortname=self.shortname,
            description=self.description,
            value=self.value,
            grid=self.grid,
            unit=self.unit,
            min=self.min,
            max=self.max,
            dtype=self.dtype,
            smooth=self.smooth,
            free=self.free,
            shared=self.shared,
            base=self.base,
        )

    def set(self, min: float | None = None, max: float | None = None) -> None:
        """Sets the limits of the parameters."""
        self.min, self.max = min, max
