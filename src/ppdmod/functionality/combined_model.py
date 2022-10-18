import numpy as np
import astropy.units as u

from typing import List
from astropy.units import Quantity

from .utils import IterNamespace, make_fixed_params, make_delta_component,\
    make_ring_component, _make_params, _calculate_sublimation_temperature,\
    temperature_gradient, optical_depth_gradient, flux_per_pixel
from .plotting_utils import plot
from ..model_components import DeltaComponent, GaussComponent,\
    RingComponent, UniformDiskComponent


# TODO: Make this class combine individual models
# TODO: Add FFT to this class
# TODO: Implement fitting to DataHandler
# TODO: Add refactor from long line of priors to model, components, len-wise
class CombinedModel:
    # TODO: Think of how to combine the models
    # TODO: Implement model combine with names of IterNamespace
    def __init__(self, fixed_params: IterNamespace, disc_params: IterNamespace,
                 wavelengths: List[Quantity], geometric_params: IterNamespace,
                 modulation_params: IterNamespace) -> None:
        """"""
        self._model_init_params = fixed_params[:-1]
        self.tau = fixed_params[-1]
        self._wavelengths = wavelengths
        self._disc_params = disc_params
        self.geometric_params = geometric_params
        self.modulation_params = modulation_params

        self._components_dic = {"ring": RingComponent, "delta": DeltaComponent,
                                "gauss": GaussComponent,
                                "uniform_disk": UniformDiskComponent}

        self._components = []
        self._components_attrs = []

        self._stellar_flux_func = None
        self._inner_radius = None

    @property
    def components(self):
        """Initialises the model's components"""
        if self._components:
            components = [component(*self._model_init_params)\
                          for component in self._components]
            return components
        else:
            raise ValueError("Add components before accessing the class's functions!")

    @property
    def pixel_scaling(self):
        if self._model_init_params.pixel_sampling is None:
            return self._model_init_params.fov/self._model_init_params.image_size
        return self._model_init_params.fov/self._model_init_params.pixel_sampling

    @property
    def inner_temperature(self):
        """Gets the inner temperature according to the radius"""
        if self._inner_radius is not None:
            return _calculate_sublimation_temperature(self._inner_radius,
                                                      self._model_init_params.distance,
                                                      self._model_init_params.lum_star)
        else:
            return self._model_init_params.sub_temp

    def _set_ones(self, image: Quantity) -> Quantity:
        """Sets and image grid to all ones"""
        image[image != 0.] = 1.*image.unit
        return image

    def add_component(self, value: IterNamespace) -> None:
        """Adds components to the model"""
        self._components.append(self._components_dic[value.component])
        self._components_attrs.append(value)

    def eval_model(self) -> Quantity:
        """Evaluates the model's radius"""
        image = None
        for i, component in enumerate(self.components):
            component_attrs = self._components_attrs[i]
            if component_attrs.component == "delta":
                self._stellar_flux_func = component.eval_flux
                continue
            if ("axis_ratio" in component_attrs.params._fields)\
                    and (self.geometric_params):
                component_attrs.params.axis_ratio = self.geometric_params.axis_ratio
                component_attrs.params.pa = self.geometric_params.pa

            # NOTE: Mention that the order has to be kept correctly and add sublimation
            # calculation here radius
            if "inner_radius" in component_attrs.params._fields:
                if self._inner_radius is None:
                    if component_attrs.params.inner_radius.value != 0.:
                        self._inner_radius = component_attrs.params.inner_radius
                    # TODO: Add sublimation radius calculation here
                    else:
                        ...
                else:
                    if self._inner_radius.value > component_attrs.params.inner_radius.value:
                        self._inner_radius = component_attrs.params.inner_radius

            temp_image = component.eval_model(component_attrs.params)

            # TODO: Check if commutative azimuthal modulation
            if self.modulation_params:
                mod_amp = self.modulation_params.mod_amp
                mod_angle = self.modulation_params.mod_angle
                temp_image = component._set_azimuthal_modulation(temp_image, mod_amp,
                                                                 mod_angle)
            if component_attrs.mod_params:
                mod_angle, mod_amp = component_attrs.mod_params.mod_angle,\
                    component_attrs.mod_params.mod_amp
                temp_image = component._set_azimuthal_modulation(temp_image, mod_amp,
                                                                 mod_angle)

            if image is None:
                image = temp_image
            else:
                image += temp_image
        return image

    def eval_object(self) -> Quantity:
        return self._set_ones(self.eval_model()).value*u.dimensionless_unscaled

    def eval_flux(self, wavelength: Quantity) -> Quantity:
        """Evaluates the flux for model"""
        # TODO: Implement stellar_flux_func here
        image = self.eval_model()
        temperature = temperature_gradient(image, self._disc_params.q,
                                           self._inner_radius, self.inner_temperature)

        optical_depth = optical_depth_gradient(image, self._disc_params.p,
                                               self._inner_radius, self.tau)
        flux = flux_per_pixel(wavelength, temperature, optical_depth, self.pixel_scaling)
        flux[flux == np.inf] = 0.*u.Jy
        if self._stellar_flux_func is not None:
            flux += self._stellar_flux_func(wavelength)
        return flux

    def eval_total_flux(self, wavelength: Quantity) -> Quantity:
        """Sums up the flux from the individual pixel [astropy.units.Jy/px] brightness
        distribution to the complete brightness [astropy.units.Jy]"""
        return np.sum(self.eval_flux(wavelength))

    # TODO: Add functionality here
    def plot(self, image: Quantity) -> None:
        plot(image)


if __name__ == "__main__":
    fixed_params = make_fixed_params(30, 128, 1500, 7900, 140, 19, 1)
    disc_params = _make_params([1., 1.],
                               [u.dimensionless_unscaled, u.dimensionless_unscaled],
                               ["q", "p"])
    geometric_params = _make_params([0.5, 140], [u.dimensionless_unscaled, u.deg],
                                    ["axis_ratio", "pa"])
    modulation_params = _make_params([0.5, 140], [u.dimensionless_unscaled, u.deg],
                                     ["mod_amp", "mod_angle"])
    wavelengths = [8*u.um]
    complete_ring = make_ring_component("inner_ring",
                                        params=[0., 0., 5., 0.])
    delta_component = make_delta_component("star")

    model = CombinedModel(fixed_params, disc_params, wavelengths,
                          geometric_params, modulation_params)
    model.add_component(complete_ring)
    model.add_component(delta_component)

