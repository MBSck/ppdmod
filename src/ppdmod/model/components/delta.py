import astropy.units as u
import astropy.constants as c

from astropy.units import Quantity

from .model_component import ModelComponent
from ...utils.general import stellar_flux, _set_zeros


# TODO: Write tests for this as well
class DeltaComponent(ModelComponent):
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
        self._component_name = "delta"

    def eval_flux(self, wavelength: Quantity) -> Quantity:
        flux = stellar_flux(wavelength, self.fixed_params.eff_temp,
                            self.fixed_params.distance, self.fixed_params.lum_star)
        return self.eval_model().value*flux

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
        image = _set_zeros(self._set_grid(), rvalue=True)
        image[self.image_centre] = 1.
        return image*u.mas


if __name__ == "__main__":
    image_size = u.Quantity(128, unit=u.dimensionless_unscaled, dtype=int)
    delta = DeltaComponent(50*u.mas, image_size, 1500*u.K,
                           7900*u.K, 140*u.pc, 19*c.L_sun, image_size)
