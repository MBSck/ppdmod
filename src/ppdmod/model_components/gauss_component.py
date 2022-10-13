import inspect
import numpy as np
import astropy.units as u

from astropy.units import Quantity

from ..functionality.model import Model
from ..functionality.utils import IterNamespace, check_and_convert


class GaussComponent(Model):
    """Two dimensional Gauss model, FFT is also Gauss

    ...

    Attributes
    ----------
        Methods
    -------
    eval_model():
        Evaluates the model
    eval_vis2():
        Evaluates the visibilities of the model
    """
    def __init__(self, *args):
        super().__init__(*args)
        self.name = "Gaussian"

    def eval_model(self, params: IterNamespace) -> Quantity:
        """Evaluates the model's radius

        Parameters
        ----------

        Returns
        --------
        model: astropy.units.Quantity
        """
        try:
            attributes = ["fwhm"]
            units = [u.mas]
            params = check_and_convert(params, attributes, units)
            self.params_count = self.non_zero_params_count = len(attributes)
        except TypeError:
            raise IOError(f"{self.name}.{inspect.stack()[0][3]}():\n"
                          f"Check input arguments! They must be: {attributes}.")

        image = self._set_grid()

        return (1/(np.sqrt(np.pi/(4*np.log(2)))*params.fwhm))\
            *np.exp((-4*np.log(2)*image**2)/params.fwhm**2)

    # def eval_vis(self, theta: np.ndarray, sampling: int,
                 # wavelength: float, size: Optional[int] = 200,
                 # uvcoords: Optional[np.ndarray] = None) -> np.array:
        # """Evaluates the visibilities of the model

        # Parameters
        # ----------
        # fwhm: int | float
            # The diameter of the sphere
        # wavelength: int
            # The sampling wavelength
        # sampling: int, optional
            # The sampling of the uv-plane
        # size: int
            # The size of the (u,v)-axis
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
            # fwhm = mas2rad(theta[0])
        # except:
            # raise IOError(f"{self.name}.{inspect.stack()[0][3]}():"
                          # " Check input arguments, theta must be"
                          # " of the form [fwhm]")

        # self._sampling = sampling
        # B, self._axis_vis  = set_uvcoords(wavelength, sampling, size,
                                          # uvcoords=uvcoords)

        # return np.exp(-(np.pi*fwhm*B)**2/(4*np.log(2)))

if __name__ == "__main__":
    gauss = GaussComponent(10, 128, 1500, 7900, 140, 19)
    params = [4*u.mas]
    gauss.plot(gauss.eval_model(params))

