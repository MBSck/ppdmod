from os import name
import numpy as np
import astropy.units as u

from astropy.units import Quantity
from typing import List
from collections import namedtuple

from .utils import make_fixed_params, make_delta_component, make_ring_component,\
    make_priors, make_params_from_priors
from ..model_components import DeltaComponent, GaussComponent,\
    RingComponent, UniformDiskComponent


# TODO: Make this class combine individual models
# TODO: Add FFT to this class
# TODO: Implement fitting to DataHandler
# TODO: Add refactor from long line of priors to model, components, len-wise
class CombinedModel:
    # TODO: Think of how to combine the models
    # TODO: Implement model combine with names of namedtuple
    def __init__(self, fixed_params: namedtuple,
                 disc_params: namedtuple,
                 selected_wavelengths: List[float],
                 geometric_priors: List[List[float]] = None) -> None:
        self._fixed_params = fixed_params
        self._disc_params = disc_params
        self._sel_wl = selected_wavelengths
        self._geometric_priors = geometric_priors

        self._components_dic = {"ring": RingComponent, "delta": DeltaComponent,
                                "gauss": GaussComponent,
                                "uniform_disk": UniformDiskComponent}

        self._components = []
        self._components_attrs = []

        self._stellar_flux_func = None

    @property
    def components(self):
        """Initialises the model's components"""
        if self._components:
            return [component(*self._fixed_params) for component in self._components]
        else:
            raise ValueError("Add components before accessing the class's functions!")

    @property
    def geometric_priors(self):
        """Sets the geometric priors"""
        if self._geometric_priors:
            units = [u.dimensionless_unscaled, u.deg]
            labels = ["axis_ratio", "pa"]
            return make_priors(self._geometric_priors, units, labels)
        else:
            raise ValueError("Set geometric priors before accessing the class's functions!")

    @property
    def geometric_params(self):
        """Gets the geometric params from the priors"""
        labels = ["axis_ratio", "pa"]
        return make_params_from_priors(self.geometric_priors, labels)

    # TODO: Write function that checks if the geometric params are input, if yes only do
    # one prior to the prior list
    # TODO: Complete all these funcitons
    def _refactor_priors_for_emcee(self):
        ...

    # TODO: Complete all these funcitons
    def _refactor_params_for_emcee(self):
        ...

    def _reset_tuple(self, tuple: namedtuple) -> namedtuple:


    def add_component(self, value: namedtuple) -> None:
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
            temp_image = component.eval_model(component_attrs.params)
            if image is None:
                image = temp_image
            else:
                image += temp_image
        return image
    def eval_flux(self) -> Quantity:
        """Evaluates the flux for model"""
        # TODO: Implement stellar_flux_func here
        ...

    def eval_total_flux(self) -> Quantity:
        """Sums up the flux from the individual pixel [astropy.units.Jy/px] brightness
        distribution to the complete brightness [astropy.units.Jy]"""
        return np.sum(self.eval_flux())

    def plot(self, image: Quantity) -> None:
        self.components[0].plot(image)



if __name__ == "__main__":
    # TODO: Make ring component maker
    fixed_params = make_fixed_params(30, 128, 1500, 7900, 140, 19, None)
    disc_params = 0
    sel_wl = [3.2]
    inner_ring_component = make_ring_component("inner_ring",
                                               [[0., 1.], [0, 180], [0., 0.], [3., 5.]])
    outer_ring_component = make_ring_component("outer_ring",
                                               [[0., 1.], [0, 180], [11., 15.], [0., 0.]])

    delta_component = make_delta_component("delta")
    geometric_priors = [[0., 1.], [0, 180]]
    model = CombinedModel(fixed_params, disc_params, sel_wl, geometric_priors)
    model.add_component(inner_ring_component)
    model.add_component(outer_ring_component)
    model.plot(model.eval_model())
