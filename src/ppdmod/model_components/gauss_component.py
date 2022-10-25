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
        self.component_name = "gauss"

    def eval_model(self, params: IterNamespace) -> Quantity:
        """Evaluates the model's radius

        Parameters
        ----------

        Returns
        --------
        model: astropy.units.Quantity
        """
        image = self._set_grid()
        return (1/(np.sqrt(np.pi/(4*np.log(2)))*params.fwhm))\
            *np.exp((-4*np.log(2)*image**2)/params.fwhm**2)


if __name__ == "__main__":
    gauss = GaussComponent(10, 128, 1500, 7900, 140, 19)
    params = [4*u.mas]
    gauss.plot(gauss.eval_model(params))

