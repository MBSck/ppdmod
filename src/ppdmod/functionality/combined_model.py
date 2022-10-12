import numpy as np

from astropy.units import Quantity


# TODO: Make this class combine individual models
# TODO: Add FFT to this class
# TODO: Implement fitting to DataHandler
# TODO: Add refactor from long line of priors to model, components, len-wise
class CombinedModel:
    # TODO: Think of how to combine the models
    # TODO: Implement model combine with names of namedtuple
    def __init__(self) -> None:
        self._components = []

    def add_component(self, component) -> None:
        self._components.append(component)

    def get_component_flux(self) -> Quantity:
        ...

    def _refactor_priors_for_emcee(self):
        ...

    def _refactor_params_for_emcee(self):
        ...

    def eval_total_flux(self, *args) -> Quantity:
        """Sums up the flux from the individual pixel [astropy.units.Jy/px] brightness
        distribution to the complete brightness [astropy.units.Jy]"""
        return np.sum(self.get_flux(*args))


