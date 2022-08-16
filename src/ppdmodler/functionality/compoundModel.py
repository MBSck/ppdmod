import numpy as np

class CompoundModel:
    """Adds 2D rings up/integrates them to create new models (e.g., a uniform
    disk or a ring structure with or without inclination)

    ...

    Methods
    -------
    integrate_rings():
        This adds the rings up to various models and shapes
        ...
    integrate_rings_vis():
        ...
    """
    def __init__(self, components: List = []):
        self.name = "Compound-Model"
        self.components = components

    def get_flux(self):
        total_flux = 0
        for c in self.components:
            total_flux += c.get_flux()

        return total_flux



if __name__ == "__main__":
    ...
