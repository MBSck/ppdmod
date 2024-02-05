import sys
from typing import Optional, Dict, List

import astropy.units as u
import numpy as np
from astropy.modeling import models

from .component import Component, AnalyticalComponent, \
        HankelComponent
from .parameter import Parameter
from .options import STANDARD_PARAMETERS, OPTIONS
from .utils import distance_to_angular


class Star(AnalyticalComponent):
    """Star defined as an analytical component.

    Parameters
    ----------
    x : int
        x pos of the component [mas].
    y : int
        y pos of the component [mas].
    dist : float
        Distance to the star.
    eff_temp : float
        The star's temperature.
    eff_radius : float
        The star's radius.

    Attributes
    ----------
    name : str
        The component's name.
    shortname : str
        The component's short name.
    description : str
        The component's description.
    params : dict of Parameter
    stellar_radius_angular : u.mas
    _image : numpy.ndarray
    """
    name = "Star"
    shortname = "St"
    description = "The flux of a star."

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._stellar_angular_radius = None

        self.params["f"] = Parameter(**STANDARD_PARAMETERS["f"])
        self.params["dist"] = Parameter(**STANDARD_PARAMETERS["dist"])
        self.params["eff_temp"] = Parameter(**STANDARD_PARAMETERS["eff_temp"])
        self.params["eff_radius"] = Parameter(**STANDARD_PARAMETERS["eff_radius"])
        self._eval(**kwargs)

    @property
    def stellar_radius_angular(self) -> u.mas:
        r"""Calculates the parallax from the stellar radius and the distance to
        the object.

        Returns
        -------
        stellar_radius_angular : astropy.units.mas
            The parallax of the stellar radius.
        """
        self._stellar_angular_radius = distance_to_angular(
            self.params["eff_radius"](), self.params["dist"]())
        return self._stellar_angular_radius

    def calculate_flux(self, wavelength: u.um) -> u.Jy:
        """Calculates the flux of the star."""
        if self.params["f"].value is not None:
            stellar_flux = self.params["f"](wavelength)
        else:
            plancks_law = models.BlackBody(
                    temperature=self.params["eff_temp"]())
            spectral_radiance = plancks_law(wavelength).to(
                u.erg/(u.cm**2*u.Hz*u.s*u.rad**2))
            stellar_flux = np.pi*(spectral_radiance
                                  * self.stellar_radius_angular**2).to(u.Jy)
        stellar_flux = stellar_flux.astype(OPTIONS.data.dtype.real)
        return stellar_flux.reshape((wavelength.size, 1))

    def _image_function(self, xx: u.mas, yy: u.mas,
                        wavelength: Optional[u.Quantity[u.m]] = None,
                        ) -> Optional[u.Quantity]:
        """Calculates the image from a 2D grid.

        Parameters
        ----------
        xx : u.mas
        yy : u.mas
        wavelength : u.m, optional

        Returns
        -------
        image : astropy.units.Quantity, optional
        """
        image = np.zeros((wavelength.size, *xx.shape))*u.Jy
        centre = xx.shape[0]//2
        star_flux = (self.calculate_flux(wavelength)/4)[..., np.newaxis]
        image[:, centre-1:centre+1, centre-1:centre+1] = star_flux
        return image.astype(OPTIONS.data.dtype.real)

    def _visibility_function(self, dim: int, pixel_size: u.mas,
                             wavelength: Optional[u.Quantity[u.um]] = None
                             ) -> np.ndarray:
        """The component's _visibility_function."""
        star_flux = np.ones((dim, dim))*self.calculate_flux(wavelength)
        return star_flux.value.astype(OPTIONS.data.dtype.real)


class GreyBody(HankelComponent):
    """An analytical implementation of an asymmetric temperature
    gradient."""
    name = "Asymmetric Continuum Grey Body"
    shortname = "AsymContinuumGreyBody"
    const_temperature = True
    continuum_contribution = True


class AsymmetricGreyBody(HankelComponent):
    """An analytical implementation of an asymmetric temperature
    gradient."""
    name = "Asymmetric Continuum Grey Body"
    shortname = "AsymContinuumGreyBody"
    asymmetric = True
    const_temperature = True
    continuum_contribution = True


class TempGradient(HankelComponent):
    """An analytical implementation of an asymmetric temperature
    gradient."""
    name = "Asymmetric Continuum Grey Body"
    shortname = "AsymContinuumGreyBody"
    const_temperature = False
    continuum_contribution = True


class AsymmetricTempGradient(HankelComponent):
    """An analytical implementation of an asymmetric temperature
    gradient."""
    name = "Asymmetric Continuum Grey Body"
    shortname = "AsymContinuumGreyBody"
    asymmetric = True
    const_temperature = False
    continuum_contribution = True


def assemble_components(
        parameters: Dict[str, Dict],
        shared_params: Optional[Dict[str, Parameter]] = None
        ) -> List[Component]:
    """Assembles a model from a dictionary of parameters."""
    components = []
    for (component, params) in parameters:
        comp = getattr(sys.modules[__name__], component)
        components.append(comp(**params, **shared_params,
                               **OPTIONS.model.constant_params))
    return components
