import inspect
import astropy.units as u
import astropy.constants as c
import matplotlib.pyplot as plt

# from scipy.special import j0
from astropy.units import Quantity
from typing import List, Union

from ..functionality.model import Model
from ..functionality.utils import IterNamespace, check_and_convert, rebin_image

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
        self.component_name = "ring"

    def eval_model(self, params: Union[IterNamespace, List]) -> Quantity:
        """Evaluates the model's radius

        Parameters
        ----------

        Returns
        --------
        image: astropy.units.Quantity
            The image's radius [astropy.units.mas]
        """
        try:
            attributes = ["axis_ratio", "pa", "inner_radius", "outer_radius"]
            units = [u.dimensionless_unscaled, u.deg, u.mas, u.mas]
            params = check_and_convert(params, attributes, units)
            self.params_count = len(attributes)
            self.non_zero_params_count = 2
        except TypeError:
            raise IOError(f"{self.name}.{inspect.stack()[0][3]}():\n"
                          f"Check input arguments! They must be: {attributes}.")

        image = self._set_grid([params.axis_ratio, params.pa])

        if params.inner_radius != 0:
            image[image < params.inner_radius] = 0.*u.mas
            self.non_zero_params_count += 1

        if params.outer_radius != 0.:
            image[image > params.outer_radius] = 0.*u.mas
            self.non_zero_params_count += 1
        return image

    # def eval_vis(self, theta: List, sampling: int,
                 # wavelength: float, size: Optional[int] = 200,
                 # incline_params: Optional[List] = [],
                 # uvcoords: np.ndarray = None) -> np.array:
        # """Evaluates the visibilities of the model

        # Parameters
        # ----------
        # r_max: int | float
            # The radius of the ring,  input in mas
        # sampling: int
            # The pixel sampling
        # wavelength: float
            # The wavelength
        # size: int, optional
            # The size of the (u,v)-plane
        # incline_params: List, optional
            # A list containing the [pos_angle, axis_ratio, inc_angle]
        # uvcoords: np.ndarray, optional

        # Returns
        # -------
        # fft: np.array
            # The analytical FFT of a ring

        # See also
        # --------
        # set_uvcoords()
        # """
        # try:
            # r_max = mas2rad(theta[0])
        # except Exception as e:
            # raise IOError(f"{self.name}.{inspect.stack()[0][3]}():"
                          # " Check input arguments, theta must be of the"
                          # " form [r_max]")

        # self._sampling, self._wavelength = sampling, wavelength
        # B, self._axis_vis = set_uvcoords(wavelength, sampling, size,
                                         # uvcoords=uvcoords, B=False)

        # return j0(2*np.pi*r_max*B)

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

