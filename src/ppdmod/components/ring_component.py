import astropy.units as u
import astropy.constants as c
import matplotlib.pyplot as plt

from astropy.units import Quantity
from typing import List, Union

from ..libs.model import Model
from ..libs.utils import IterNamespace, rebin_image

# TODO: Make function that automatically greps the docstrings of functions that need to be
# implemented


class RingComponent(Model):
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

    def eval_model(self, params: Union[IterNamespace, List]) -> Quantity:
        """Evaluates the model's radius

        Parameters
        ----------
        params: IterNamespace
            An IterNamespace containing the information for the 'axis_ratio', the
            'positional_angle', the 'inner_radius' and optionally the 'outer_radius'

        Returns
        --------
        image: astropy.units.Quantity
            The image's radius [astropy.units.mas]
        """
        image = self._set_grid([params.axis_ratio, params.pa])

        if params.inner_radius != 0.:
            image[image < params.inner_radius] = 0.

        if params.outer_radius != 0.:
            image[image > params.outer_radius] = 0.
        return image


if __name__ == "__main__":
    params_model = [10*u.mas, u.Quantity(128, unit=u.dimensionless_unscaled, dtype=int),
                    1500*u.K, 7900*u.K, 140*u.pc, 19*c.L_sun,
                    u.Quantity(4096, unit=u.dimensionless_unscaled, dtype=int)]
    ring = RingComponent(*params_model)
    params_model2 = params_model.copy()
    params_model2[-1] = u.Quantity(128, unit=u.dimensionless_unscaled, dtype=int)
    ring2 = RingComponent(*params_model2)
    params = [0.5*u.dimensionless_unscaled, 145*u.deg, 1*u.mas, 0*u.mas]
    image = ring.eval_model(params)
    image_low_res_no_rebin = ring2.eval_model(params)
    image_rebin, factor = rebin_image(image, (128, 128), rfactor=True)
    print(factor)
    fig, (ax, bx, cx) = plt.subplots(1, 3)
    ax.imshow(image.value)
    bx.imshow(image_rebin.value)
    cx.imshow(image_low_res_no_rebin.value)
    plt.show()

