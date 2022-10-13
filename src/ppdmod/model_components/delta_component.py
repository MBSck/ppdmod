import numpy as np
import astropy.units as u

from astropy.units import Quantity

from ..functionality.model import Model
from ..functionality.utils import stellar_flux

# TODO: Write tests for this as well
class DeltaComponent(Model):
    """Delta function/Point source model

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
        self.name = "Delta"

    def eval_flux(self, wavelength: Quantity) -> Quantity:
        return self.eval_model()\
            .value*stellar_flux(wavelength, self.luminosity_star,
                                self.effective_temperature, self.distance)

    def eval_model(self) -> Quantity:
        """Evaluates the model

        Parameters
        ----------
        mas_size: int
            The size of the FOV
        px_size: int
            The size of the model image

        Returns
        --------
        model: np.array
        """
        image = self._set_zeros(self._set_grid())
        image[self.image_centre] = 1.*u.mas
        return image

    # def eval_vis(self, theta: List, sampling: int) -> np.array:
        # """Evaluates the visibilities of the model

        # Parameters
        # ----------
        # flux: float
            # The flux of the object
        # sampling: int
            # The sampling of the uv-plane

        # Returns
        # -------
        # visibility: np.array
        # """
        # try:
            # flux = float(theta[0])
        # except:
            # raise IOError(f"{self.name}.{inspect.stack()[0][3]}():"
                          # " Check input arguments, theta must"
                          # " be of the form [flux]")

        # self._sampling = self._size = sampling

        # return flux*np.ones((sampling, sampling))

if __name__ == "__main__":
    delta = DeltaComponent(50, 128, 1500, 7900, 140, 19)
    delta.plot(delta.eval_flux(8*u.um))

