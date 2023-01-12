import numpy as np
import astropy.units as u

from astropy.units import Quantity
from typing import Tuple, List, Optional

from .utils import IterNamespace, make_fixed_params, _set_ones, _make_axis

# TODO: Make sure all is tested! Write test for set_ones and improve all tests
# TODO: Make good docstrings
class Model:
    """Model baseclass """
    def __init__(self, fixed_params: IterNamespace) -> None:
        self.fixed_params = fixed_params
        self.pixel_size = self.fixed_params.fov/self.fixed_params.image_size
        self._component_name = None
        self._polar_angle = None

    @property
    def image_centre(self) -> Tuple:
        """Returns a tuple containing the indicies for the centre of the the model image

        Returns
        -------
        image_centre: Tuple[astropy.units.Quantity]
            The model image's centre [astropy.units.dimensionless_unscaled]
        """
        if self._component_name == "delta":
            return (self.fixed_params.image_size//2, self.fixed_params.image_size//2)
        else:
            return (self.fixed_params.pixel_sampling//2,
                    self.fixed_params.pixel_sampling//2)

    def _set_grid(self, incline_params: Optional[List[float]] = None) -> Quantity:
        """Sets the size of the model and its centre. Returns the polar coordinates

        Parameters
        ----------
        incline_params: List[float], optional
            A list of the inclination parameters [axis_ratio, pos_angle]
            [astropy.units.dimensionless_unscaled, astropy.units.deg]
            DISCLAIMER: The axis_ratio should be in [0., 1.] and the pos_angle in [0, 180]

        Returns
        -------
        radius: np.array
            The radius [astropy.units.mas/px]
        """
        # TODO: Does center shift, xc, yc need to be applied?
        if self._component_name == "delta":
            x = _make_axis(self.fixed_params.image_size//2, self.fixed_params.image_size)
        else:
            x = _make_axis(self.fixed_params.image_size//2,
                           self.fixed_params.pixel_sampling)

        x *= self.pixel_size
        y = x[:, np.newaxis]

        if incline_params is not None:
            axis_ratio, pos_angle = incline_params
            pos_angle = pos_angle.to(u.rad)
            xr, yr = (x*np.cos(pos_angle)-y*np.sin(pos_angle))/axis_ratio,\
                x*np.sin(pos_angle)+y*np.cos(pos_angle)
            self._polar_angle = np.arctan2(xr, yr)
            radius = np.sqrt(xr**2+yr**2)
        else:
            self._polar_angle = np.arctan2(x, y)
            radius = np.sqrt(x**2+y**2)
        return radius

    def _set_azimuthal_modulation(self, image: Quantity,
                                  amplitude: Quantity,
                                  modulation_angle: Quantity) -> Quantity:
        """Calculates the azimuthal modulation of the object

        Parameters
        ----------
        image: astropy.units.Quantity
            The model's image [astropy.units.Jy/px] or any image to modulate
        amplitude: Quantity
            The 'c'-amplitude [astropy.units.dimensionless_unscaled]
        polar_angle: astropy.units.Quantity
            The polar angle of the x, y-coordinates [astropy.units.rad]

        Returns
        -------
        azimuthal_modulation: astropy.units.Quantity
            The azimuthal modulation [astropy.units.dimensionless_unscaled]
        """
        # TODO: Implement Modulation field like Jozsef?
        modulation_angle = modulation_angle.to(u.rad)
        total_mod = amplitude*np.cos(self._polar_angle-modulation_angle)
        image *= 1.*total_mod.unit + total_mod
        image.value[image.value < 0.] = 0.
        return image

    def eval_object(self, params: IterNamespace) -> Quantity:
        return _set_ones(self.eval_model(params), rvalue=True)*u.dimensionless_unscaled


if __name__ == "__main__":
    fixed_params = make_fixed_params(50, 128, 1500, 7900, 140, 19)
    model = Model(fixed_params)
    print(model._set_grid())
