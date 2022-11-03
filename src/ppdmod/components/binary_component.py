import numpy as np
import astropy.units as u

from typing import Union
from astropy.units import Quantity

from ..lib.model import Model
from ..lib.utils import IterNamespace


class BinaryComponent(Model):
    """..."""
    def __init__(self, *args):
        super().__init__(*args)
        self.component_name = "binary"
        self._pos_star1, self._pos_star2 = None, None
        self._flux1, self._flux2 = None, None
        self._image_seperation = None

    def _flux_per_pixel(self) -> Quantity:
        if (self._flux1 is None) or (self._flux2 is None):
            raise RuntimeError("Before determining the flux, first evaluate the Binary model!")
        flux = self._set_zeros(self._set_grid()).value*u.Jy
        flux[self.pos_star1] = self._flux1
        flux[self.pos_star2] = self._flux2
        return flux

    # TODO: Make this work as a realistic component
    def eval_model(self, params: Union[IterNamespace, List]) -> Quantity:
        """Evaluates the model

        Binary values have to be input in negative for some values

        Parameters
        ----------

        Returns
        --------
        model: np.array

        See also
        --------
        set_size()
        """
        image = self._set_zeros(self._set_grid()/u.mas)
        self._image_seperation = np.sqrt((params.x1-params.x2)**2 + \
                                         (params.y1-params.y2)**2)*self.pixel_scaling

        # FIXME: Fix this mess at some point
        self.pos_star1 = (self.image_centre +\
                          u.Quantity((params.y1, params.x1), dtype=int))
        self.pos_star2 = (self.image_centre +\
                          u.Quantity((params.y2, params.x2), dtype=int))
        self._flux1, self._flux2 = params.flux1, params.flux2
        image[self.pos_star1] = 1*u.dimensionless_unscaled
        image[self.pos_star2] = 1*u.dimensionless_unscaled
        return image


if __name__ == "__main__":
    binary = BinaryComponent(50, 128, 1500, 7900, 140, 19)
    coords = [10, -10, 15, -15]*u.dimensionless_unscaled
    theta = [10*u.Jy, 10*u.Jy]
    theta.extend(coords)
    binary.plot(binary.eval_model(theta))
