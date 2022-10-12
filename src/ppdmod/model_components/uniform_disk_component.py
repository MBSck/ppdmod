import inspect
import numpy as np
import astropy.units as u

from scipy.special import j1
from astropy.units import Quantity
from typing import List, Optional
from collections import namedtuple

from ..functionality.model import Model
from ..functionality.utils import _check_and_convert

# TODO: Implement analytical formula for inclined disk and uniform one in same eval_vis
# TODO: Implement inclination in eval model

class UniformDiskComponent(Model):
    """Uniformly bright disc model

    ...

    Methods
    -------
    eval_model():
        Evaluates the model
    eval_vis2():
        Evaluates the visibilities of the model
    """
    def __init__(self, *args):
        super().__init__(*args)
        self.name = "Uniform Disk"

    def eval_model(self, params: namedtuple) -> Quantity:
        """Evaluates the model

        Parameters
        ----------

        Returns
        --------
        image: np.ndarray
            The image of the model

        See also
        --------
        set_size()
        """
        # FIXME: There is a 0 in the middle of the uniform disk
        try:
            attributes = ["axis_ratio", "pa", "diameter"]
            units = [u.dimensionless_unscaled, u.deg, u.mas]
            params = _check_and_convert(params, attributes, units)
        except TypeError:
            raise IOError(f"{self.name}.{inspect.stack()[0][3]}():\n"
                          f"Check input arguments! They must be: {attributes}.")

        image = self._set_grid([params.axis_ratio, params.pa])
        image[image > params.diameter/2] = 0.*u.mas
        return image

    def eval_object(self, params: namedtuple) -> Quantity:
        return self._set_ones(self.eval_model(params)).value*u.dimensionless_unscaled

    # def eval_vis(self, params: named, uvcoords: np.ndarray = None) -> Quantity:
        # """Evaluates the visibilities of the model

        # Parameters
        # ---------
        # diameter: int
            # The diameter of the sphere
        # sampling: int
            # The sampling of the uv-plane
        # wavelength: float
            # The sampling wavelength
        # uvcoords: List[float], optional
            # If uv-coords are given, then the visibilities are calculated for
            # precisely these.

        # Returns
        # -------
        # visibility: np.array

        # See also
        # --------
        # set_uvcoords()
        # """
        # try:
            # diameter = mas2rad(theta[0])
        # except:
            # raise IOError(f"{self.name}.{inspect.stack()[0][3]}():"
                          # " Check input arguments, theta must be of"
                          # " the form [diameter]")

        # self._sampling, self._wavelength = sampling, wavelength
        # B, self._axis_vis = set_uvcoords(wavelength, sampling, size,
                                         # uvcoords=uvcoords, B=False)
        # return 2*j1(np.pi*diameter*B)/(np.pi*diameter*B)

if __name__ == "__main__":
    uniform_disk = UniformDiskComponent(10, 128, 1500, 7900, 140, 19)
    params = [0.5*u.dimensionless_unscaled, 145*u.deg, 3*u.mas]
    uniform_disk.plot(uniform_disk.eval_object(params))

