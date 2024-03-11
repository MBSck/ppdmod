from typing import Optional, Tuple

import astropy.units as u
import numpy as np

from .parameter import Parameter
from .options import STANDARD_PARAMETERS, OPTIONS
from .utils import compute_effective_baselines, broadcast_baselines


class Component:
    """The base class for the component.

    Parameters
    ----------
    xx : float
        The x-coordinate of the component.
    yy : float
        The x-coordinate of the component.
    dim : float
        The dimension [px].
    """
    name = "Generic component"
    shortname = "GenComp"
    description = "This is the class from which all components are derived."
    _elliptic = False

    def __init__(self, **kwargs):
        """The class's constructor."""
        self.x = Parameter(**STANDARD_PARAMETERS.x)
        self.y = Parameter(**STANDARD_PARAMETERS.y)
        self.dim = Parameter(**STANDARD_PARAMETERS.dim)
        self.pa = Parameter(**STANDARD_PARAMETERS.pa)
        self.inc = Parameter(**STANDARD_PARAMETERS.inc)

        if not self.elliptic:
            self.inc.free = self.pa.free = False
        self.eval(**kwargs)

    @property
    def elliptic(self) -> bool:
        """Gets if the component is elliptic."""
        return self._elliptic

    @elliptic.setter
    def elliptic(self, value: bool) -> None:
        """Sets the position angle and the parameters to free or false
        if elliptic is set."""
        self._elliptic = self.inc.free = self.pa.free = value

    def eval(self, **kwargs):
        """Sets the parameters (values) from the keyword arguments."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                if key in ["inc", "pa"]:
                    self.elliptic = True
                if isinstance(value, Parameter):
                    setattr(self, key, value)
                else:
                    getattr(self, key).value = value


    def get_params(self, free: Optional[bool] = False):
        """Gets all the parameters of a component.

        Parameters
        ----------
        component : Component
            The component for which the parameters should be fetched.
        only_free : bool, optional
            If only the free parameters should be returned, by default False

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
                params[attribute] = value
        return params

    def compute_internal_grid(
            self, dim: int, pixel_size: u.mas
            ) -> Tuple[u.Quantity[u.mas], u.Quantity[u.mas]]:
        """Calculates the model grid.

        Parameters
        ----------
        dim : float, optional
        pixel_size : float, optional

        Returns
        -------
        xx : astropy.units.mas
            The x-coordinate grid.
        yy : astropy.units.mas
            The y-coordinate grid.
        """
        return np.array([])*u.mas, np.array([])*u.mas

    def translate_image_func(
            self, xx: u.mas, yy: u.mas
            ) -> Tuple[u.Quantity[u.mas], u.Quantity[u.mas]]:
        """Shifts the coordinates in image space according to an offset."""
        xx, yy = map(lambda x: u.Quantity(value=x, unit=u.mas), [xx, yy])
        xx, yy =  xx-self.x(), yy-self.y()
        return xx.astype(OPTIONS.data.dtype.real), yy.astype(OPTIONS.data.dtype.real)

    # TODO: Check if the positive factor in the exp here is correct?
    def translate_vis_func(self, baselines: 1/u.rad, baseline_angles: u.rad) -> np.ndarray:
        """Translates a coordinate shift in image space to Fourier space."""
        translation = np.exp(2*1j*np.pi*baselines*(self.x()*np.cos(baseline_angles)+self.y()*np.sin(baseline_angles)))
        return translation.value.astype(OPTIONS.data.dtype.complex)

    def flux_func(self, wavelength: u.um) -> np.ndarray:
        """Calculates the total flux from the hankel transformation."""
        return np.array([]).astype(OPTIONS.data.dtype.real)

    def vis_func(self, baselines: 1/u.rad, baseline_angles: u.rad,
                 wavelength: u.um, **kwargs) -> np.ndarray:
        """Computes the correlated fluxes."""
        return np.array([]).astype(OPTIONS.data.dtype.complex)

    def compute_flux(self, wavelength: u.um) -> np.ndarray:
        """Computes the total fluxes."""
        return np.abs(self.flux_func(wavelength)).astype(OPTIONS.data.dtype.real)

    def compute_complex_vis(self, ucoord: u.m, vcoord: u.m,
                            wavelength: u.um, **kwargs) -> np.ndarray:
        """Computes the correlated fluxes."""
        baselines, baseline_angles = compute_effective_baselines(
                ucoord, vcoord, self.inc(), self.pa())
        wavelength, baselines, baseline_angles = broadcast_baselines(
                wavelength, baselines, baseline_angles, ucoord)
        vis = self.vis_func(baselines, baseline_angles, wavelength, **kwargs)
        vis = vis.squeeze(-1) if vis.shape[-1] == 1 else vis
        shift = self.translate_vis_func(baselines, baseline_angles)
        shift = shift.squeeze(-1) if shift.shape[-1] == 1 else shift
        return (vis*shift).astype(OPTIONS.data.dtype.complex)

    def image_func(self, xx: u.mas, yy: u.mas, wavelength: u.um) -> np.ndarray:
        """Calculates the image."""
        return np.array([]).astype(OPTIONS.data.dtype.real)

    def compute_image(self, dim: int, pixel_size: u.mas,
                      wavelength: u.um) -> np.ndarray:
        """Computes the image."""
        wavelength = wavelength[:, np.newaxis, np.newaxis]
        pixel_size = pixel_size if isinstance(pixel_size, u.Quantity)\
                else pixel_size*u.mas
        xx = np.linspace(-0.5, 0.5, dim)*dim*pixel_size
        xx, yy = self.translate_image_func(*np.meshgrid(xx, xx))

        if self.elliptic:
            xx = xx*np.cos(self.pa())-yy*np.sin(self.pa())*self.inc()
            yy = xx*np.sin(self.pa())+yy*np.cos(self.pa())

        image = self.image_func(xx, yy, wavelength)
        return image.astype(OPTIONS.data.dtype.real)


# TODO: Think about the inclination and elongation for this component
class Convolver(Component):
    """A class that enables the convolution of multiple components.

    Parameters
    ----------
    comp1 : Component
        The first component.
    comp2 : Component
        The second component.
    """
    name = "Convolver"
    shortname = "Conv"
    description = "This a class enabling the convolution of multiple components."

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.eval(**kwargs)

    def eval(self, **kwargs):
        """Sets the components from the keyword arguments."""
        self.components = {}
        for key, value in kwargs.items():
            if isinstance(value, Component):
                setattr(self, key, value)
                self.components[key] = value

    def vis_func(self, baselines: 1/u.rad, baseline_angles: u.rad,
                 wavelength: u.um, **kwargs) -> np.ndarray:
        """Computes the correlated fluxes via the hankel transformation."""
        vis = [comp.vis_func(baselines, baseline_angles, wavelength, **kwargs) for comp in self.components.values()]
        vis = [v.squeeze(-1) if v.shape[-1] == 1 else v for v in vis]
        return np.prod(vis, axis=0).astype(OPTIONS.data.dtype.complex)
