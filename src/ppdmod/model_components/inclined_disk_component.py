import inspect
import numpy as np

from typing import List
from astropy.units import Quantity
from scipy.special import j0
from collections import namedtuple

from ..functionality.model import Model

class InclinedDiskComponent(Model):
    """By a certain position angle inclined disk

    ...

    Methods
    -------
    eval_model():
        Evaluates the model
    eval_vis():
        Evaluates the visibilities of the model
    """

    def __init__(self, *args):
        super().__init__(*args)
        self.name = "Inclined Disk"

    def eval_model(self) -> None:
        """Evaluates the model. In case of zero divison error, the major will be replaced by 1

        Parameters
        ----------
        pass

        Returns
        --------
        model: np.array
        """
        raise RuntimeError("Not implemented! Use 'ring_component' instead!")

    def eval_vis(self, params: namedtuple,
                 wavelength: float) -> Quantity:
        """Evaluates the visibilities of the model

        Parameters
        ----------
        r_max: int | float
            The max radius of the inclined disk
        r_0: int | float
            The inital radius of the inclined disk
        T_0: int
            The temperature at the initial radius r_0
        distance: int
            The object's distance from the observer
        pos_angle_ellipsis: float
        pos_angle_axis: float
        inc_angle: float
            The inclination angle of the disk
        wavelength: float
            The sampling wavelength
        sampling: int
            The sampling of the uv-plane

        Returns
        -------
        visibility: np.array

        See also
        --------
        set_uvcoords()
        """
        # TODO: Implement the analytical formula
        ...

if __name__ == "__main__":
    ...

