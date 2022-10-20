import inspect
import numpy as np
import astropy.units as u

from astropy.units import Quantity
from typing import Tuple, List, Optional

from .utils import _set_ones

# TODO: Make sure all is tested! Write test for set_ones and improve all tests
# TODO: Make good docstrings
class Model:
    """Model

    ...

    Methods
    -------
    eval_model():
        Evaluates the model
    eval_vis2():
        Evaluates the visibilities of the model
    """
    def __init__(self, field_of_view: Quantity, image_size: int,
                 sublimation_temperature: Quantity,
                 effective_temperature: Quantity,
                 distance: Quantity, luminosity_star: Quantity,
                 pixel_sampling: Quantity) -> None:
        """"""
        self.field_of_view = field_of_view
        self.image_size = image_size
        self.sublimation_temperature = sublimation_temperature
        self.effective_temperature = effective_temperature
        self.luminosity_star = luminosity_star
        self.distance = distance
        self.pixel_sampling = pixel_sampling

        self.component_name = None
        # TODO: Check why this is a thing? The axes...
        self.axes_image, self.axes_complex_image = [], []
        self.polar_angle = np.array([])

    @property
    def image_centre(self) -> Tuple:
        """Returns index"""
        if self.component_name == "delta":
            return (self.image_size//2, self.image_size//2)
        else:
            return (self.pixel_sampling//2, self.pixel_sampling//2)

    @property
    def pixel_scaling(self):
        return self.field_of_view/self.image_size

    def _set_grid(self, incline_params: Optional[List[float]] = None) -> Quantity:
        """Sets the size of the model and its centre. Returns the polar coordinates

        Parameters
        ----------
        mas_size: int
            Sets the size of the image [astropy.units.mas]
        size: int
            Sets the range of the model image and implicitly the x-, y-axis.
            Size change for simple models functions like zero-padding
        pixel_sampling: int, optional
            The pixel sampling [px]
        incline_params: List[float], optional
            A list of the inclination parameters [axis_ratio, pos_angle]
            [None, astropy.units.rad]

        Returns
        -------
        radius: np.array
            The radius [astropy.units.mas/px]
        """
        # TODO: Does center shift, xc, yc need to be applied?
        if self.component_name == "delta":
            x = np.linspace(-self.image_size//2, self.image_size//2,
                            self.image_size, endpoint=False)*self.pixel_scaling
        else:
            x = np.linspace(-self.image_size//2, self.image_size//2,
                            self.pixel_sampling, endpoint=False)*self.pixel_scaling
        y = x[:, np.newaxis]

        if incline_params:
            try:
                axis_ratio, pos_angle = incline_params
            except:
                raise IOError(f"{inspect.stack()[0][3]}(): Check input"
                              " arguments, 'incline_params' must be of the"
                              " form [axis_ratio, pos_angle]")

            if not isinstance(axis_ratio, u.Quantity):
                axis_ratio *= u.dimensionless_unscaled
            elif axis_ratio.unit != u.dimensionless_unscaled:
                raise IOError("Enter the axis ratio in"\
                              " [astropy.units.dimensionless_unscaled] or unitless!")

            if not isinstance(pos_angle, u.Quantity):
                pos_angle *= u.deg
            elif pos_angle.unit != u.deg:
                raise IOError("Enter the positional angle in [astropy.units.deg] or"\
                              " unitless!")

            if (axis_ratio.value < 0.) or (axis_ratio.value > 1.):
                raise ValueError("The axis ratio must be between [0., 1.]")

            if (pos_angle.value < 0) or (pos_angle.value > 180):
                raise ValueError("The positional angle must be between [0, 180]")

            pos_angle = pos_angle.to(u.rad)

            # NOTE: This was taken from Jozsef's code and until here output is equal
            xr, yr = (x*np.cos(pos_angle)-y*np.sin(pos_angle))/axis_ratio,\
                x*np.sin(pos_angle)+y*np.cos(pos_angle)
            radius = np.sqrt(xr**2+yr**2)
            self.axes_image, self.polar_angle = [xr, yr], np.arctan2(xr, yr)
        else:
            radius = np.sqrt(x**2+y**2)
            self.axes_image, self.polar_angle = [x, y], np.arctan2(x, y)

        return radius

    # TODO: Rework this function alike the others
    # def _set_uv_grid(self, wavelength: float,
                     # incline_params: List[float] = None,
                     # uvcoords: np.ndarray = None,
                     # vector: Optional[bool] = True) -> Quantity:
        # """Sets the uv coords for visibility modelling

        # Parameters
        # ----------
        # incline_params: List[float], optional
            # A list of the three angles [axis_ratio, pos_angle, inc_angle]
        # uvcoords: List[float], optional
            # If uv-coords are given, then the visibilities are calculated for them
        # vector: bool, optional
            # Returns the baseline vector if toggled true, else the baselines

        # Returns
        # -------
        # baselines: astropy.units.Quantity
            # The baselines for the uvcoords [astropy.units.m]
        # uvcoords: astropy.units.Quantity
            # The axis used to calculate the baselines [astropy.units.m]
        # """
        # if not isinstance(wavelength, u.Quantity):
            # wavelength *= u.um
        # elif wavelength.unit != u.um:
            # raise IOError("Enter the wavelength in [astropy.units.um] or unitless!")

        # if uvcoords is None:
            # axis = np.linspace(-self.image_size, size, sampling, endpoint=False)*u.m

            # # Star overhead sin(theta_0)=1 position
            # u, v = axis/wavelength.to(u.m),\
                # axis[:, np.newaxis]/wavelength.to(u.m)

        # else:
            # axis = uvcoords/wavelength.to(u.m)
            # u, v = np.array([uvcoord[0] for uvcoord in uvcoords]), \
                    # np.array([uvcoord[1] for uvcoord in uvcoords])

        # if angles is not None:
            # try:
                # if len(angles) == 2:
                    # axis_ratio, pos_angle = incline_params
                # else:
                    # axis_ratio = incline_params[0]
                    # pos_angle, inc_angle = map(lambda x: (x*u.deg).to(u.rad),
                                               # incline_params[1:])

                # u_inclined, v_inclined = u*np.cos(pos_angle)+v*np.sin(pos_angle),\
                        # v*np.cos(pos_angle)-u*np.sin(pos_angle)

                # if len(angles) > 2:
                    # v_inclined = v_inclined*np.cos(inc_angle)

                # baselines = np.sqrt(u_inclined**2+v_inclined**2)
                # baseline_vector = baselines*wavelength.to(u.m)
                # self.axes_complex_image = [u_inclined, v_inclined]
            # except:
                # raise IOError(f"{inspect.stack()[0][3]}(): Check input"
                              # " arguments, ellipsis_angles must be of the form"
                              # " either [pos_angle] or "
                              # " [ellipsis_angle, pos_angle, inc_angle]")

        # else:
            # baselines = np.sqrt(u**2+v**2)
            # baseline_vector = baselines*wavelength.to(u.m)
            # self.axes_complex_image = [u, v]

        # return baseline_vector if vector else baselines

    def _set_azimuthal_modulation(self, image: Quantity,
                                  amplitude: Quantity,
                                  modulation_angle: Quantity) -> Quantity:
        """Calculates the azimuthal modulation of the object

        Parameters
        ----------
        image: astropy.units.Quantity
            The model's image [astropy.units.Jy/px]
        amplitude: Quantity
            The 'c'-amplitude [astropy.units.dimensionless_unscaled]
        polar_angle: astropy.units.Quantity
            The polar angle of the x, y-coordinates [astropy.units.rad]

        Returns
        -------
        azimuthal_modulation: astropy.units.Quantity
            The azimuthal modulation [astropy.units.dimensionless_unscaled]
        """
        if not isinstance(amplitude, u.Quantity):
            amplitude *= u.dimensionless_unscaled
        elif amplitude.unit != u.dimensionless_unscaled:
            raise IOError("Enter the modulation amplitude in [astropy.units.deg]"\
                          " or unitless!")

        if not isinstance(modulation_angle, u.Quantity):
            modulation_angle *= u.deg
        elif modulation_angle.unit != u.deg:
            raise IOError("Enter the modulation angle in [astropy.units.deg]"\
                          " or unitless!")

        # TODO: Implement Modulation field like Jozsef?
        modulation_angle = modulation_angle.to(u.rad)
        if self.polar_angle.size == 0:
            raise ValueError("Evaluate the model or create a grid to access the"\
                             " polar angle!")
        else:
            total_mod = amplitude*np.cos(self.polar_angle-modulation_angle)
        image *= 1 + total_mod
        image.value[image.value < 0.] = 0.
        return image

    def eval_model(self) -> Quantity:
        """Evaluates the model image's radius

        Returns
        --------
        image: Quantity
            A two-dimensional model image [astropy.units.mas]
        """
        pass

    def eval_object(self) -> Quantity:
        """Evaluates the model image's object

        Returns
        --------
        image: Quantity
            A two-dimensional model image [astropy.units.dimensionless_unscaled]
        """
        return _set_ones(self.eval_model(params)).value*u.dimensionless_unscaled

    def eval_flux(self) -> Quantity:
        """Evaluates the complex visibility function of the model.

        Returns
        -------
        complex_visibility_function: Quantity
            A two-dimensional complex visibility function [astropy.units.m]
        """
        pass


if __name__ == "__main__":
    ...
