import copy
from typing import Tuple

import astropy.units as u
import numpy as np

from .options import OPTIONS
from .parameter import Parameter
from .utils import broadcast_baselines, compute_effective_baselines


# TODO: Implement automated assertion to make sure the parameters are the same and all
class Component:
    """The base class for the component."""

    name = "GenComp"
    label = None
    description = "This is base component are derived."

    def eval(self, **kwargs) -> None:
        """Sets the parameters (values) from the keyword arguments."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                if isinstance(value, Parameter):
                    setattr(self, key, value.copy())
                else:
                    if isinstance(getattr(self, key), Parameter):
                        getattr(self, key).value = value
                    else:
                        setattr(self, key, value)

    def copy(self) -> "Component":
        """Copies the component."""
        return copy.deepcopy(self)

    def get_params(self, free: bool = False, shared: bool = False) -> dict:
        """Gets all the parameters of a component.

        Parameters
        ----------
        component : Component
            The component for which the parameters should be fetched.
        free : bool, optional
            If only the free parameters should be returned, by default False.
        shared : bool, optional
            If only the shared parameters should be returned, by default False.

        Returns
        -------
        params : dict of Parameter
        """
        params = {}
        for attribute in dir(self):
            value = getattr(self, attribute)
            if isinstance(value, Parameter):
                if (free and not value.free) or (free and value.shared):
                    continue

                if shared and not value.shared:
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

    name = "FourierComp"
    description = "The component from which all analytical components are derived."
    _asymmetric = False

    def __init__(self, **kwargs):
        """The class's constructor."""
        self.fr = Parameter(base="fr")
        self.x = Parameter(base="x")
        self.y = Parameter(base="y")
        self.pa = Parameter(base="pa")
        self.cinc = Parameter(base="cinc")
        self.dim = Parameter(base="dim")

        # TODO: Switch this to a more intiutive modulation (in polar coordinates)
        for i in range(1, OPTIONS.model.modulation + 1):
            rho_str, theta_str = f"rho{i}", f"theta{i}"
            rho = Parameter(name=rho_str, free=self.asymmetric, base="rho")
            theta = Parameter(name=theta_str, free=self.asymmetric, base="theta")
            setattr(self, rho_str, rho)
            setattr(self, theta_str, theta)

    @property
    def asymmetric(self) -> bool:
        """Gets if the component is asymmetric."""
        return self._asymmetric

    @asymmetric.setter
    def asymmetric(self, value: bool) -> None:
        """Sets the position angle and the parameters to free or false
        if asymmetry is set."""
        self._asymmetric = value
        for i in range(1, OPTIONS.model.modulation + 1):
            getattr(self, f"rho{i}").free = value
            getattr(self, f"theta{i}").free = value

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

    # TODO: Check this again
    def translate_vis_func(
        self, baselines: 1 / u.rad, baseline_angles: u.rad
    ) -> np.ndarray:
        """Translates a coordinate shift in image space to Fourier space."""
        uv_coords = np.exp((1j * self.x().to(u.rad) * np.cos(baseline_angles)).value) \
            * np.exp((1j * self.y().to(u.rad) * np.sin(baseline_angles)).value)
        translation = np.exp(2j * np.pi * baselines * np.angle(uv_coords) * u.rad)
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
            ucoord, vcoord, self.cinc(), self.pa()
        )
        wavelength, baselines, baseline_angles = broadcast_baselines(
            wavelength, baselines, baseline_angles, ucoord
        )

        vis = self.vis_func(baselines, baseline_angles, wavelength, **kwargs)
        vis = vis.reshape(vis.shape[:-1]) if vis.shape[-1] == 1 else vis
        shift = self.translate_vis_func(baselines, baseline_angles)
        shift = shift.reshape(shift.shape[:-1]) if shift.shape[-1] == 1 else shift

        if self.name != "Point":
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

        pa_rad = self.pa().to(u.rad)
        xr = xx * np.cos(pa_rad) - yy * np.sin(pa_rad)
        yr = xx * np.sin(pa_rad) + yy * np.cos(pa_rad)
        xx, yy = xr * (1 / self.cinc()), yr

        image = self.image_func(xx, yy, pixel_size, wavelength)
        return (self.fr() * image).value.astype(OPTIONS.data.dtype.real)
