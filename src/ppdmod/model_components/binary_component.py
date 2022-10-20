import inspect
import numpy as np
import astropy.units as u

from typing import Union
from astropy.units import Quantity

from ..functionality.model import Model
from ..functionality.utils import IterNamespace


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
        try:
            attributes = ["flux1", "flux2", "x1", "x2", "y1", "y2"]
            coord_units = [u.dimensionless_unscaled for _ in range(4)]
            units = [u.Jy, u.Jy]
            units.extend(coord_units)
            params = _check_and_convert(params, attributes, units)
            self.params_count = len(attributes)
        except TypeError:
            raise IOError(f"{self.name}.{inspect.stack()[0][3]}():\n"
                          f"Check input arguments! They must be: {attributes}.")

        image = self._set_zeros(self._set_grid()/u.mas)

        self._image_seperation = np.sqrt((params.x1-params.x2)**2 + \
                                         (params.y1-params.y2)**2)*self.pixel_scaling

        # FIXME: Fix this mess at some point
        self.pos_star1 = (self.image_centre + u.Quantity((params.y1+params.x1, dtype=int))
        self.pos_star2 = (self.image_centre + u.Quantity((params.y2,params.x2, dtype=int))
        self._flux1, self._flux2 = params.flux1, params.flux2
        image[self.pos_star1] = 1*u.dimensionless_unscaled
        image[self.pos_star2] = 1*u.dimensionless_unscaled
        return image

    # def eval_vis(self, theta: List, wavelength: float,
                 # sampling: int, size: Optional[int] = 200,
                 # uvcoords: np.ndarray = None) -> np.array:
        # """Evaluates the visibilities of the model

        # Parameters
        # ----------
        # theta: List
            # The parameters for the model/analytical function
        # wavelength: float
            # The sampling wavelength
        # sampling: int
            # The pixel sampling
        # size: int, optional
            # Sets the range of the (u,v)-plane in meters, with size being the
            # longest baseline uvcoords: List[float], optional
            # If uv-coords are given, then the visibilities are calculated for
            # precisely these.

        # Returns
        # -------
        # complex_visibilities: np.array
            # The Fourier transform of the intensity distribution

        # See also
        # --------
        # set_uvcoords()
        # """
        # try:
            # flux1, flux2, x1, y1, x2, y2 = theta
        # except:
            # raise IOError(f"{self.name}.{inspect.stack()[0][3]}():"
                          # " Check input arguments, theta must be of"
                          # " the form [flux1, flux2, separation]")

        # sep_vec = np.array([x1-x2, y1-y2])
        # x1, y1, x2, y2 = map(lambda x: mas2rad(x*self.pixel_scale), [x1, y1, x2, y2])

        # B, self._axis_vis = set_uvcoords(wavelength, sampling, size, B=True)

        # global axis1, axis2
        # u, v = axis1, axis2 = self._axis_vis
        # flux1_contribution = flux1*np.exp(2*np.pi*-1j*(u*x1+v*y1))
        # flux2_contribution = flux2*np.exp(2*np.pi*-1j*(u*x2+v*y2))

        # return flux1_contribution + flux2_contribution


if __name__ == "__main__":
    binary = BinaryComponent(50, 128, 1500, 7900, 140, 19)
    coords = [10, -10, 15, -15]*u.dimensionless_unscaled
    theta = [10*u.Jy, 10*u.Jy]
    theta.extend(coords)
    binary.plot(binary.eval_model(theta))
