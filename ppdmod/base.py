import copy
from typing import Tuple

import astropy.units as u
import numpy as np

from .options import OPTIONS
from .parameter import Parameter
from .utils import (
    transform_coordinates,
    translate_image_func,
    translate_vis_func,
)


class Component:
    """The base class for the component."""

    name = "GenComp"
    label = None
    description = "This is base component are derived."

    def __init__(self, **kwargs):
        """The class's constructor."""
        self.flux_lnf = Parameter(name="flux_lnf", base="lnf")
        self.t3_lnf = Parameter(name="t3_lnf", base="lnf")
        self.vis_lnf = Parameter(name="vis_lnf", base="lnf")

    def eval(self, **kwargs) -> None:
        """Sets the parameters (values) from the keyword arguments."""
        for key, val in kwargs.items():
            if hasattr(self, key):
                if isinstance(val, Parameter):
                    setattr(self, key, val.copy())
                else:
                    if isinstance(getattr(self, key), Parameter):
                        getattr(self, key).value = val
                    else:
                        setattr(self, key, val)

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
            If the free parameters should be returned, by default False.
        shared : bool, optional
            If the shared parameters should be returned, by default False.

        Returns
        -------
        params : dict of Parameter
        """
        params = {}
        for attribute in dir(self):
            value = getattr(self, attribute)
            if isinstance(value, Parameter):
                if shared and free:
                    if not (value.shared and value.free):
                        continue
                elif free:
                    if not value.free or value.shared:
                        continue
                elif shared:
                    if not value.shared or value.free:
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
        super().__init__(**kwargs)
        self.fr = Parameter(base="fr")
        self.r = Parameter(base="r")
        self.phi = Parameter(base="phi")
        self.pa = Parameter(base="pa")
        self.cinc = Parameter(base="cinc")
        self.dim = Parameter(base="dim")

        for i in range(1, OPTIONS.model.modulation + 1):
            rho_str, theta_str = f"rho{i}", f"theta{i}"
            rho = Parameter(name=rho_str, free=self.asymmetric, base="rho")
            theta = Parameter(name=theta_str, free=self.asymmetric, base="theta")
            setattr(self, rho_str, rho)
            setattr(self, theta_str, theta)

    @property
    def x(self) -> u.Quantity:
        r = self.r()
        if self.r.unit == u.au:
            r = (r.to(u.au) / self.dist().to(u.pc)).value * 1e3 * u.mas
        return r * np.sin(self.phi().to(u.rad))

    @property
    def y(self) -> u.Quantity:
        r = self.r()
        if self.r.unit != u.mas:
            r = (r.to(u.au) / self.dist().to(u.pc)).value * 1e3 * u.mas
        return r * np.cos(self.phi().to(u.rad))

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

    def compute_internal_grid( self) -> Tuple[u.Quantity[u.au], u.Quantity[u.au]]:
        """Calculates the model grid.

        Parameters
        ----------

        Returns
        -------
        xx : astropy.units.au
            The x-coordinate grid.
        yy : astropy.units.au
            The y-coordinate grid.
        """
        return np.array([]) * u.au, np.array([]) * u.au

    def vis_func(self, spf: 1 / u.rad, psi: u.rad, wl: u.um, **kwargs) -> np.ndarray:
        """Computes the correlated fluxes."""
        return np.array([]).astype(OPTIONS.data.dtype.complex)

    def compute_complex_vis(
        self, ucoord: u.m, vcoord: u.m, wl: u.um, **kwargs
    ) -> np.ndarray:
        """Computes the correlated fluxes."""
        ut, vt = transform_coordinates(ucoord, vcoord, self.cinc(), self.pa())
        wl = wl.reshape(-1, 1)
        utb = (ut / wl.to(u.m)).value[..., np.newaxis] / u.rad
        vtb = (vt / wl.to(u.m)).value[..., np.newaxis] / u.rad
        spf, psi = np.hypot(utb, vtb), np.arctan2(utb, vtb)

        shift = translate_vis_func(
            utb.value, vtb.value, self.x.to(u.rad).value, self.y.to(u.rad).value
        )
        shift = shift.reshape(shift.shape[:-1]) if shift.shape[-1] == 1 else shift
        vis = self.vis_func(spf, psi, wl, **kwargs)
        vis = vis.reshape(vis.shape[:-1]) if vis.shape[-1] == 1 else vis

        if self.name != "Point":
            vis *= self.fr(wl).value

        return (vis * shift).astype(OPTIONS.data.dtype.complex)

    def image_func(
        self, xx: u.mas, yy: u.mas, pixel_size: u.mas, wavelength: u.um
    ) -> np.ndarray:
        """Calculates the image."""
        return np.array([]).astype(OPTIONS.data.dtype.real)

    def compute_image(self, dim: int, pixel_size: u.mas, wl: u.um) -> np.ndarray:
        """Computes the image."""
        wl = wl[np.newaxis, np.newaxis]
        xx = np.linspace(-0.5, 0.5, dim) * pixel_size * dim
        xxt, yyt = transform_coordinates(*np.meshgrid(xx, xx), self.cinc(), self.pa(), axis="x")
        xxs, yys = translate_image_func(xxt, yyt, self.x, self.y)
        image = self.image_func(xxs, yys, pixel_size, wl)
        return (self.fr(wl) * image).value.astype(OPTIONS.data.dtype.real)
