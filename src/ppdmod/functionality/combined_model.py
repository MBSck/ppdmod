import numpy as np

from astropy.units import Quantity


# TODO: Make this class combine individual models
class CombinedModel:
    # TODO: Think of how to combine the models
    def __init__(self) -> None:
        self._components = []

    def add_component(self, component) -> None:
        self._components.append(component)

    def get_component_flux(self) -> Quantity:
        ...

    def total_flux(self, *args) -> Quantity:
        """Sums up the flux from the individual pixel [astropy.units.Jy/px] brightness
        distribution to the complete brightness [astropy.units.Jy]"""
        return np.sum(self.get_flux(*args))


