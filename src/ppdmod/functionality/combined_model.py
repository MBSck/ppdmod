import numpy as np
import astropy.units as u

from astropy.units import Quantity
from typing import List, Optional

from .utils import IterNamespace, make_fixed_params, make_delta_component,\
    make_ring_component, _make_priors, _make_params_from_priors, make_disc_params,\
    _calculate_sublimation_temperature, temperature_gradient, optical_depth_gradient,\
    flux_per_pixel
from .fourier import FastFourierTransform
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
    def __init__(self, fixed_params: IterNamespace,
                 disc_params: IterNamespace,
                 wavelength: List[Quantity],
                 global_geometric_priors: List[List[float]] = None,
                 global_modulation_priors: List[List[float]] = None,
                 zero_padding_order: Optional[int] = 1) -> None:
        """"""
        self._model_init_params = fixed_params[:-1]
        self.tau = fixed_params[-1]
        self._wl = wavelength
        self._disc_params = disc_params
        self._geometric_priors = global_geometric_priors
        self._mod_priors = global_modulation_priors
        self.zp_order = zero_padding_order

        self._mod_params = self._set_mod_params()
        self._geometric_params = self._set_geometric_params()

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
    def fourier(self):
        return FastFourierTransform(self.eval_flux(self._wl), self._wl,
                                    self.pixel_scaling, self.zp_order)

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

    def _set_geometric_priors(self):
        """Sets the geometric priors"""
        if self._geometric_priors is not None:
            units = [u.dimensionless_unscaled, u.deg]
            labels = ["axis_ratio", "pa"]
            return _make_priors(self._geometric_priors, units, labels)
        else:
            return False

    def _set_geometric_params(self):
        """Gets the geometric params from the priors"""
        labels = ["axis_ratio", "pa"]
        self._geometric_priors = self._set_geometric_priors()
        return _make_params_from_priors(self._geometric_priors, labels)\
            if self._geometric_priors else False

    def _set_mod_priors(self):
        """Sets the modulation priors"""
        if self._mod_priors is not None:
            units = [u.deg, u.dimensionless_unscaled]
            labels = ["mod_angle", "mod_amp"]
            return _make_priors(self._mod_priors, units, labels)
        else:
            return False

    def _set_mod_params(self):
        """Gets the modulation params from the priors"""
        labels = ["mod_angle", "mod_amp"]
        self._mod_priors = self._set_mod_priors()
        return _make_params_from_priors(self._mod_priors, labels)\
            if self._mod_priors else False

    # TODO: Write function that checks if the geometric params are input, if yes only do
    # one prior to the prior list
    # TODO: Complete all these funcitons
    def _refactor_priors_for_emcee(self):
        ...

    # TODO: Complete all these funcitons
    def _refactor_params_for_emcee(self):
        ...

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
                    and (self._geometric_params):
                component_attrs.params.axis_ratio = self._geometric_params.axis_ratio
                component_attrs.params.pa = self._geometric_params.pa

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
            # FIXME: Azimuthal modulation is broken fix
            if self._mod_priors:
                component_attrs.mod_priors = self._mod_priors
                component_attrs.mod_params = self._mod_params
            elif component_attrs.mod_priors:
                mod_angle, mod_amp = component_attrs.mod_params.mod_angle,\
                    component_attrs.mod_params.mod_amp
                temp_image = component._set_azimuthal_modulation(temp_image, mod_amp,
                                                                 mod_angle)

            if image is None:
                image = temp_image
            else:
                image += temp_image
        return image

    def eval_flux(self, wavelength: Quantity) -> Quantity:
        """Evaluates the flux for model"""
        # TODO: Implement stellar_flux_func here
        image = self.eval_model()
        temperature = temperature_gradient(image, self._disc_params.params.q,
                                           self._inner_radius, self.inner_temperature)

        optical_depth = optical_depth_gradient(image, self._disc_params.params.p,
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

    def get_interpolated_amp_and_phase(self, uv_coords: Quantity,
                                       uv_coords_cphase: Quantity) -> List[Quantity]:
        return self.fourier.get_uv2fft2(uv_coords, uv_coords_cphase)

    # TODO: Add functionality here
    def plot(self, image: Quantity) -> None:
        plot(image)

    def get_amp_phase(self) -> None:
        return self.fourier.get_amp_phase()

    def plot_amp_and_phase(self, matplot_axis: Optional[List] = [],
                           zoom: Optional[int] = 500,
                           uv_coords: Optional[List] = None,
                           uv_coords_cphase: Optional[List] = None,
                           plt_save: Optional[bool] = False) -> None:
        self.fourier.plot_amp_phase(matplot_axis, zoom, uv_coords,
                                    uv_coords_cphase, plt_save)


if __name__ == "__main__":
    # TODO: Make ring component maker
    fixed_params = make_fixed_params(30, 128, 1500, 7900, 140, 19, 1)
    disc_params = make_disc_params([[0., 1.], [0., 1.]])
    wavelength = 8*u.um
    complete_ring = make_ring_component("inner_ring",
                                        [[0., 0.], [0, 0], [3., 5.], [0., 0.]])
    # inner_ring_component = make_ring_component("inner_ring",
                                               # [[0., 0.], [0, 0], [3., 5.], [5., 6.]])
    # outer_ring_component = make_ring_component("outer_ring",
                                               # [[0., 0.], [0, 0], [6., 8.], [0., 0.]])
    delta_component = make_delta_component("star")

    global_geometric_priors = [[0., 1.], [0, 180]]
    global_modulation_priors = [[0., 1.], [0, 360]]
    model = CombinedModel(fixed_params, disc_params,
                          wavelength, global_geometric_priors,
                          global_modulation_priors, zero_padding_order=2)
    model.add_component(complete_ring)
    # model.add_component(inner_ring_component)
    # model.add_component(outer_ring_component)
    model.add_component(delta_component)
    print(model.fourier.dim, model.fourier.freq_axis)
    print(model.eval_total_flux(wavelength))
    model.plot_amp_and_phase(zoom=1000)
