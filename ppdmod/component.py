from typing import Tuple

import astropy.units as u
import numpy as np

from .parameter import Parameter
from .options import STANDARD_PARAMETERS, OPTIONS


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
        self.params = {}
        self.params["x"] = Parameter(**STANDARD_PARAMETERS.x)
        self.params["y"] = Parameter(**STANDARD_PARAMETERS.y)
        self.params["dim"] = Parameter(**STANDARD_PARAMETERS.dim)
        self.params["pixel_size"] = Parameter(
                **STANDARD_PARAMETERS.pixel_size)
        self.params["pa"] = Parameter(**STANDARD_PARAMETERS.pa)
        self.params["elong"] = Parameter(**STANDARD_PARAMETERS.elong)

        if not self.elliptic:
            self.params["pa"].free = False
            self.params["elong"].free = False
        self._eval(**kwargs)

    @property
    def elliptic(self) -> bool:
        """Gets if the component is elliptic."""
        return self._elliptic

    @elliptic.setter
    def elliptic(self, value: bool) -> None:
        """Sets the position angle and the parameters to free or false
        if elliptic is set."""
        if value:
            self.params["pa"].free = True
            self.params["elong"].free = True
        else:
            self.params["pa"].free = False
            self.params["elong"].free = False
        self._elliptic = value

    def _eval(self, **kwargs):
        """Sets the parameters (values) from the keyword arguments."""
        for key, value in kwargs.items():
            if key in self.params:
                if isinstance(value, Parameter):
                    self.params[key] = value
                else:
                    self.params[key].value = value

    def calculate_internal_grid(
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
        xx, yy =  xx-self.params["x"](), yy-self.params["y"]()
        return xx.astype(OPTIONS.data.dtype.real), yy.astype(OPTIONS.data.dtype.real)

    # TODO: Check if the positive factor in the exp here is correct?
    def translate_vis_func(
            self, ucoord: u.m, vcoord: u.m, wavelength: u.um) -> np.ndarray:
        """Translates a coordinate shift in image space to Fourier space."""
        wavelength = wavelength[:, np.newaxis]
        if ucoord.shape[0] != 1:
            wavelength = wavelength[..., np.newaxis]

        x, y = map(lambda x: self.params[x]().to(u.rad), ["x", "y"])
        ucoord, vcoord = map(lambda x: u.Quantity(x, unit=u.m), [ucoord, vcoord])
        ucoord, vcoord = map(lambda x: x/wavelength.to(u.m)/u.rad, [ucoord, vcoord])
        translation = np.exp(2*1j*np.pi*(ucoord*x+vcoord*y)).value 
        return translation.astype(OPTIONS.data.dtype.complex)

    def flux_func(self, wavelength: u.um) -> u.Jy:
        """Calculates the total flux from the hankel transformation."""
        return (np.array([])*u.Jy).astype(OPTIONS.data.dtype.real)

    def vis_func(self, ucoord: u.m, vcoord: u.m,
                 wavelength: u.um, **kwargs) -> np.ndarray:
        """Computes the correlated fluxes."""
        return np.array([]).astype(OPTIONS.data.dtype.complex)

    def compute_flux(self, wavelength: u.um) -> u.Jy:
        """Computes the total fluxes."""
        return np.abs(self.flux_func(wavelength)).astype(OPTIONS.data.dtype.real)

    def compute_vis(self, ucoord: u.m, vcoord: u.m,
                    wavelength: u.um, **kwargs):
        """Computes the correlated fluxes."""
        vis = self.vis_func(ucoord, vcoord, wavelength, **kwargs)
        vis *= self.translate_vis_func(ucoord, vcoord, wavelength)
        return vis.astype(OPTIONS.data.dtype.complex)

    def image_func(self, xx: u.mas, yy: u.mas,
                   pixel_size: u.mas, wavelength: u.um) -> u.Jy:
        """Calculates the image."""
        return (np.array([])*u.Jy).astype(OPTIONS.data.dtype.real)

    def compute_image(self, dim: int, pixel_size: u.mas, wavelength: u.um) -> u.Jy:
        """Computes the image."""
        wavelength = wavelength[:, np.newaxis, np.newaxis]
        pixel_size = pixel_size if isinstance(pixel_size, u.Quantity)\
                else pixel_size*u.mas
        xx = np.linspace(-0.5, 0.5, dim)*dim*pixel_size
        xx, yy = self.translate_image_func(*np.meshgrid(xx, xx))

        if self.elliptic:
            pa, elong = self.params["pa"](), self.params["elong"]()
            xx = xx*np.cos(pa)-yy*np.sin(pa)
            yy = (xx*np.sin(pa)+yy*np.cos(pa))*elong

        image = self.image_func(xx, yy, pixel_size, wavelength)
        return image.astype(OPTIONS.data.dtype.real)
