from typing import Any, Tuple, List, Union, Optional

import numpy as np
import astropy.units as u

from ...utils.general import IterNamespace, make_fixed_params, _set_ones, _make_axis

# TODO: Make sure all is tested! Write test for set_ones and improve all tests
# TODO: Make good docstrings
class ModelComponent:
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

    def _set_grid(self, incline_params: Optional[List[float]] = None) -> u.mas:
        """Sets the size of the model and its centre. Returns the polar coordinates

        Parameters
        ----------
        incline_params: List[float], optional
            A list of the inclination parameters [axis_ratio, pos_angle]
            [astropy.units.dimensionless_unscaled, astropy.units.deg]
            DISCLAIMER: The axis_ratio should be in [0., 1.] and the pos_angle in [0, 180]

        Returns
        -------
        radius: u.mas/px
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

    def _set_azimuthal_modulation(self, image: Union[Any, u.mas, u.Jy],
                                  amplitude: u.dimensionless_unscaled,
                                  modulation_angle: u.deg) -> u.mas:
        """Calculates the azimuthal modulation of the object

        Parameters
        ----------
        image: u.mas/px | u.Jy/px
            The model's image or any image to modulate
        amplitude: u.dimensionless_unscaled
            The 'c'-amplitude

        Returns
        -------
        azimuthally_modulated_image: u.mas
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
