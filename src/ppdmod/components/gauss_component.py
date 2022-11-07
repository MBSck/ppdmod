import numpy as np
import astropy.units as u

from astropy.units import Quantity

import matplotlib.pyplot as plt

from ..lib.model import Model
from ..lib.utils import IterNamespace, make_fixed_params, _make_params


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
        self._component_name = "gauss"

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
    fixed_params = make_fixed_params(10, 128, 1500, 7900, 140, 19)
    gauss = GaussComponent(fixed_params)
    params = _make_params([4.], [u.mas], ["fwhm"])
    plt.imshow(gauss.eval_model(params).value)
    plt.show()

