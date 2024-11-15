import copy
from typing import Tuple

import astropy.units as u
import numpy as np

from .options import OPTIONS, STANDARD_PARAMS
from .parameter import Parameter
from .utils import broadcast_baselines, compute_effective_baselines


class Component:
    """The base class for the component."""

    name = "Generic component"
    shortname = "GenComp"
    label = None
    description = "This is base component are derived."

    def eval(self, **kwargs) -> None:
        """Sets the parameters (values) from the keyword arguments."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                if isinstance(value, Parameter):
                    setattr(self, key, value)
                else:
                    if isinstance(getattr(self, key), Parameter):
                        getattr(self, key).value = value
                    else:
                        setattr(self, key, value)

    def copy(self):
        """Copies the component."""
        return copy.deepcopy(self)

    def get_params(self, free: bool = False, fixed: bool = False) -> dict:
        """Gets all the parameters of a component.

        Parameters
        ----------
        component : Component
            The component for which the parameters should be fetched.
        free : bool, optional
            If only the free parameters should be returned, by default False.
        fixed : bool, optional
            If only the fixed parameters should be returned, by default False.

        Returns
        -------
        params : dict of Parameter
        """
        params = {}
        for attribute in dir(self):
            value = getattr(self, attribute)
            if isinstance(value, Parameter):
                if free and not value.free:
                    continue
                if free and value.shared:
                    continue
                elif fixed and value.free:
                    continue

                params[attribute] = value

        return params

    def flux_func(self, wavelength: u.um) -> np.ndarray:
        """Calculates the flux."""
        return np.array([]).astype(OPTIONS.data.dtype.real)

    def compute_flux(self, wavelength: u.um) -> np.ndarray:
        """Computes the fluxes."""
        return np.abs(self.flux_func(wavelength)).astype(OPTIONS.data.dtype.real)


class FourierComponent(Component):
    """The base class for the Fourier (analytical) component.

    Parameters
    ----------
    xx : float
        The x-coordinate of the component.
    yy : float
        The x-coordinate of the component.
    dim : float
        The dimension (px).
    """

    name = "Fourier component"
    shortname = "FourierComp"
    description = "The component from which all analytical components are derived."
    _elliptic = True
    _asymmetric = False

    def __init__(self, **kwargs):
        """The class's constructor."""
        self.fr = Parameter(**STANDARD_PARAMS.fr)
        self.x = Parameter(**STANDARD_PARAMS.x)
        self.y = Parameter(**STANDARD_PARAMS.y)
        self.pa = Parameter(**STANDARD_PARAMS.pa)
        self.inc = Parameter(**STANDARD_PARAMS.inc)
        self.dim = Parameter(**STANDARD_PARAMS.dim)

        for i in range(1, OPTIONS.model.modulation + 1):
            setattr(self, f"c{i}", Parameter(**STANDARD_PARAMS.c))
            setattr(self, f"s{i}", Parameter(**STANDARD_PARAMS.s))

        if not self.elliptic:
            self.inc.free = self.pa.free = False

        for i in range(1, OPTIONS.model.modulation + 1):
            s = getattr(self, f"s{i}")
            c = getattr(self, f"c{i}")
            s.name = s.shortname = f"s{i}"
            c.name = c.shortname = f"c{i}"
            c.free = s.free = self.asymmetric

    @property
    def elliptic(self) -> bool:
        """Gets if the component is elliptic."""
        return self._elliptic

    @elliptic.setter
    def elliptic(self, value: bool) -> None:
        """Sets the position angle and the parameters to free or false
        if elliptic is set."""
        self._elliptic = self.inc.free = self.pa.free = value

    @property
    def asymmetric(self) -> bool:
        """Gets if the component is elliptic."""
        return self._asymmetric

    @asymmetric.setter
    def asymmetric(self, value: bool) -> None:
        """Sets the position angle and the parameters to free or false
        if elliptic is set."""
        self._asymmetric = value
        for i in range(1, OPTIONS.model.modulation + 1):
            s = getattr(self, f"s{i}")
            c = getattr(self, f"c{i}")
            s.free = c.free = value

    def compute_internal_grid(
        self, dim: int, pixel_size: u.au
    ) -> Tuple[u.Quantity[u.au], u.Quantity[u.au]]:
        """Calculates the model grid.

        Parameters
        ----------
        dim : float, optional
        pixel_size : float, optional

        Returns
        -------
        xx : astropy.units.au
            The x-coordinate grid.
        yy : astropy.units.au
            The y-coordinate grid.
        """
        return np.array([]) * u.au, np.array([]) * u.au

    def translate_image_func(
        self, xx: u.mas, yy: u.mas
    ) -> Tuple[u.Quantity[u.mas], u.Quantity[u.mas]]:
        """Shifts the coordinates in image space according to an offset."""
        xx, yy = map(lambda x: u.Quantity(value=x, unit=u.mas), [xx, yy])
        xx, yy = xx - self.x(), yy - self.y()
        return xx.astype(OPTIONS.data.dtype.real), yy.astype(OPTIONS.data.dtype.real)

    def translate_vis_func(
        self, baselines: 1 / u.rad, baseline_angles: u.rad
    ) -> np.ndarray:
        """Translates a coordinate shift in image space to Fourier space."""
        uv_coords = self.x() * np.cos(baseline_angles) + self.y() * np.sin(
            baseline_angles
        )
        translation = np.exp(2j * np.pi * baselines * uv_coords.to(u.rad))
        return translation.value.astype(OPTIONS.data.dtype.complex)

    def vis_func(
        self, baselines: 1 / u.rad, baseline_angles: u.rad, wavelength: u.um, **kwargs
    ) -> np.ndarray:
        """Computes the correlated fluxes."""
        return np.array([]).astype(OPTIONS.data.dtype.complex)

    def compute_complex_vis(
        self, ucoord: u.m, vcoord: u.m, wavelength: u.um, **kwargs
    ) -> np.ndarray:
        """Computes the correlated fluxes."""
        baselines, baseline_angles = compute_effective_baselines(
            ucoord, vcoord, self.inc(), self.pa()
        )
        wavelength, baselines, baseline_angles = broadcast_baselines(
            wavelength, baselines, baseline_angles, ucoord
        )

        vis = self.vis_func(baselines, baseline_angles, wavelength, **kwargs)
        vis = vis.reshape(vis.shape[:-1]) if vis.shape[-1] == 1 else vis
        shift = self.translate_vis_func(baselines, baseline_angles)
        shift = shift.reshape(shift.shape[:-1]) if shift.shape[-1] == 1 else shift

        if self.shortname != "Point":
            vis *= self.fr()

        return (vis * shift).astype(OPTIONS.data.dtype.complex)

    def image_func(
        self, xx: u.mas, yy: u.mas, pixel_size: u.mas, wavelength: u.um
    ) -> np.ndarray:
        """Calculates the image."""
        return np.array([]).astype(OPTIONS.data.dtype.real)

    def compute_image(
        self, dim: int, pixel_size: u.mas, wavelength: u.um
    ) -> np.ndarray:
        """Computes the image."""
        wavelength = (
            wavelength
            if isinstance(wavelength, u.Quantity)
            else u.Quantity(wavelength, u.um)
        )
        try:
            wavelength = wavelength[:, np.newaxis, np.newaxis]
        except TypeError:
            wavelength = wavelength[np.newaxis, np.newaxis]
        pixel_size = (
            pixel_size
            if isinstance(pixel_size, u.Quantity)
            else u.Quantity(pixel_size, u.mas)
        )

        xx = np.linspace(-0.5, 0.5, dim) * pixel_size * dim
        xx, yy = self.translate_image_func(*np.meshgrid(xx, xx))

        if self.elliptic:
            pa_rad = self.pa().to(u.rad)
            xr = xx * np.cos(pa_rad) - yy * np.sin(pa_rad)
            yr = xx * np.sin(pa_rad) + yy * np.cos(pa_rad)
            xx, yy = xr * (1 / self.inc()), yr

        image = self.image_func(xx, yy, pixel_size, wavelength)
        return (self.fr() * image).value.astype(OPTIONS.data.dtype.real)
