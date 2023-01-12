from typing import List, Union

import astropy.units as u
import astropy.constants as const
import matplotlib.pyplot as plt

from .model_component import ModelComponent
from ...utils.general import IterNamespace

# TODO: Make function that automatically greps the docstrings of functions that need to be
# implemented
class RingComponent(ModelComponent):
    """Infinitesimal thin ring model. Can be both cirular or an ellipsoid, i.e.
    inclined

    ...

    Methods
    -------
    eval_model():
        Evaluates the model
    eval_vis2():
        Evaluates the visibilities of the model

    See also
    --------
    set_grid()
    set_uvcoords()
    """
    def __init__(self, *args):
        super().__init__(*args)
        self._component_name = "ring"

    def eval_model(self, params: Union[IterNamespace, List]) -> u.mas:
        """Evaluates the model's radius

        Parameters
        ----------
        params: IterNamespace
            An IterNamespace containing the information for the 'axis_ratio', the
            'positional_angle', the 'inner_radius' and optionally the 'outer_radius'

        Returns
        --------
        image: u.mas
            The image's radius
        """
        image = self._set_grid([params.axis_ratio, params.pa])

        if params.inner_radius != 0.:
            image[image < params.inner_radius] = 0.

        if params.outer_radius != 0.:
            image[image > params.outer_radius] = 0.
        return image


if __name__ == "__main__":
    params_model = [10*u.mas, u.Quantity(128, unit=u.dimensionless_unscaled, dtype=int),
                    1500*u.K, 7900*u.K, 140*u.pc, 19*const.L_sun,
                    u.Quantity(4096, unit=u.dimensionless_unscaled, dtype=int)]
    ring = RingComponent(*params_model)
    params = [0.5*u.dimensionless_unscaled, 145*u.deg, 1*u.mas, 0*u.mas]
    image = ring.eval_model(params)
    plt.imshow(image.value)
    plt.show()
