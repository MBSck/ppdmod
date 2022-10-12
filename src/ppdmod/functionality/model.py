import inspect
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as c

from collections import namedtuple
from astropy.modeling import models
from astropy.units import Quantity
from typing import Tuple, List, Optional


# NOTE: Implement FFT as a part of the base_model_class, maybe?
# NOTE: Think about calling FFT class in model class to evaluate model

# TODO: Add checks for right unit input in all the files
# TODO: Make sure all is tested!
# TODO: Make docstrings proper
# TODO: Write test for set_ones and improve all tests


class Model:
    """Model metaclass

    ...

    Methods
    -------
    eval_model():
        Evaluates the model
    eval_vis2():
        Evaluates the visibilities of the model
    """
    # TODO: Make repr function that tells what the model has been initialised with
    def __init__(self, field_of_view: Quantity, image_size: int,
                 sublimation_temperature: int, effective_temperature: int,
                 distance: int, luminosity_star: int,
                 pixel_sampling: Optional[int] = None) -> None:
        # TODO: Maybe make a specific save name for the model also
        self.name = None
        self.axes_image, self.axes_complex_image, self.polar_angle = None, None, None

        self._field_of_view = field_of_view*u.mas
        # TODO: Make checks so only even numbers can be input for image_size
        self._image_size = u.Quantity(image_size, unit=u.dimensionless_unscaled, dtype=int)
        self._pixel_sampling = self.image_size if pixel_sampling\
            is None else pixel_sampling
        self._sublimation_temperature = sublimation_temperature*u.K
        self._effective_temperature = effective_temperature*u.K
        self._luminosity_star = luminosity_star*c.L_sun
        self._distance = distance*u.pc

        self.sublimation_radius = self._calculate_sublimation_radius()


    # TODO: Add docs and return types for properties
    @property
    def field_of_view(self):
        return self._field_of_view

    @field_of_view.setter
    def field_of_view(self, value):
        if not isinstance(value, u.Quantity):
            self._field_of_view = value*u.mas
        elif value.unit != u.mas:
            raise IOError("Wrong unit has been input, field of view needs to"\
                          " be in [astropy.units.mas] or unitless!")

    @property
    def image_size(self):
        return self._image_size

    @image_size.setter
    def image_size(self, value):
        if not isinstance(value, u.Quantity):
            self._image_size = u.Quantity(value,
                                          unit=u.dimensionless_unscaled,
                                          dtype=int)
        elif value.unit != u.dimensionless_unscaled:
            raise IOError("Wrong unit has been input, field of view needs to"\
                          " be in [astropy.units.dimensionless_unscaled] or unitless!")

    @property
    def image_centre(self) -> Tuple:
        # TODO: Add that this is index and not Quantity
        return (self.image_size//2, self.image_size//2)

    @property
    def pixel_sampling(self):
        return self._pixel_sampling

    @pixel_sampling.setter
    def pixel_sampling(self, value):
        self._pixel_sampling = value

    @property
    def pixel_scaling(self):
        return self.field_of_view/self.pixel_sampling

    @property
    def sublimation_temperature(self):
        return self._sublimation_temperature

    @sublimation_temperature.setter
    def sublimation_temperature(self, value):
        if not isinstance(value, u.Quantity):
            self._sublimation_temperature *= u.K
        elif value.unit != u.K:
            raise IOError("Wrong unit has been input, sublimation temperature needs to"\
                          " be in [astropy.units.K] or unitless!")
        self._sublimation_temperature = value

    @property
    def effective_temperature(self):
        return self._effective_temperature

    @effective_temperature.setter
    def effective_temperature(self, value):
        if not isinstance(value, u.Quantity):
            self._effective_temperature = value*u.K
        elif value.unit != u.K:
            raise IOError("Wrong unit has been input, effective temperature needs to be"\
                          " in [astropy.units.K] or unitless!")

    @property
    def luminosity_star(self):
        return self._luminosity_star

    @luminosity_star.setter
    def luminosity_star(self, value):
        if not isinstance(value, u.Quantity):
            self._luminosity_star = value*c.L_sun
        elif value.unit != u.W:
            raise IOError("Wrong unit has been input, luminosity needs to be in"\
                          "[astropy.units.W] or unitless!")

    @property
    def distance(self):
        return self._distance

    @distance.setter
    def distance(self, value):
        if not isinstance(value, u.Quantity):
            self._distance = value*u.pc
        elif value.unit != u.pc:
            raise IOError("Wrong unit has been input, distance needs to be in"\
                          "[astropy.units.pc] or unitless!")

    def _convert_orbital_radius_to_parallax(self, orbital_radius: Quantity,
                                              distance: Optional[Quantity] = None
                                              ) -> Quantity:
        """Calculates the parallax [astropy.units.mas] from the orbital radius
        [astropy.units.m]. The formula for the angular diameter is used

        Parameters
        ----------
        orbital_radius: astropy.units.Quantity
            The orbital radius [astropy.units.m]
        distance: astropy.units.Quantity
            The distance to the star from the observer [astropy.units.pc]

        Returns
        -------
        parallax: astropy.units.Quantity
            The angle of the orbital radius [astropy.units.mas]
        """
        if distance is None:
            distance = self.distance
        elif not isinstance(value, u.Quantity):
            distance *= u.pc
        elif value.unit != u.pc:
            raise IOError("Wrong unit has been input, distance needs to be in"\
                          "[astropy.units.pc] or unitless!")

        return (1*u.rad).to(u.mas)*(orbital_radius.to(u.m)/distance.to(u.m))

    def _convert_parallax_to_orbital_radius(self, parallax: Quantity,
                                              distance: Optional[Quantity] = None
                                              ) -> Quantity:
        """Calculates the orbital radius [astropy.units.m] from the parallax
        [astropy.units.mas]. The formula for the angular diameter is used

        Parameters
        ----------
        parallax: astropy.units.Quantity
            The angle of the orbital radius [astropy.units.mas]
        distance: astropy.units.Quantity
            The distance to the star from the observer [astropy.units.pc]

        Returns
        -------
        orbital_radius: astropy.units.Quantity
            The orbital radius [astropy.units.m]
        """
        if distance is None:
            distance = self.distance
        elif not isinstance(value, u.Quantity):
            distance *= u.pc
        elif value.unit != u.pc:
            raise IOError("Wrong unit has been input, distance needs to be in"\
                          "[astropy.units.pc] or unitless!")

        return (parallax*distance.to(u.m))/(1*u.rad).to(u.mas)

    def _calculate_stellar_radius(self) -> Quantity:
        """Calculates the stellar radius [astropy.units.m] from its attributes

        Returns
        -------
        stellar_radius: astropy.units.Quantity
            The star's radius [astropy.units.m]
        """
        return np.sqrt(self.luminosity_star/\
                       (4*np.pi*c.sigma_sb*self.effective_temperature**4))

    def _calculate_sublimation_temperature(self, inner_radius: Quantity) -> Quantity:
        """Calculates the sublimation temperature at the inner rim of the disc. Assumes
        input to be in [astropy.units.mas]

        Parameters
        ----------
        inner_radius: astropy.units.Quantity
            The inner radius of the disc [astropy.units.mas]

        Returns
        -------
        sublimation_temperature: astropy.units.Quantity
            The sublimation temperature [astropy.units.K]
        """
        if not isinstance(inner_radius, u.Quantity):
            inner_radius *= u.mas

        if inner_radius.unit == u.mas:
            inner_radius = self._convert_parallax_to_orbital_radius(inner_radius)
        elif inner_radius.unit != u.mas:
            raise IOError("Enter the inner radius in [astropy.units.mas] or unitless!")

        return (self.luminosity_star/(4*np.pi*c.sigma_sb*inner_radius**2))**(1/4)

    def _calculate_sublimation_radius(self,
                                      inner_temperature: Optional[Quantity] = None
                                      ) -> Quantity:
        """Calculates the sublimation radius at the inner rim of the disc

        Returns
        -------
        sublimation_radius: astropy.units.Quantity
            The sublimation radius [astropy.units.mas]
        """
        if inner_temperature is None:
            inner_temperature = self.sublimation_temperature
        elif not isinstance(inner_temperature, u.Quantity):
            inner_temperature *= u.K
        elif inner_temperature.unit != u.K:
            raise IOError("Enter the inner temperature in [astropy.units.K] or unitless!")

        radius = np.sqrt(self.luminosity_star/(4*np.pi*c.sigma_sb*inner_temperature**4))
        return self._convert_orbital_radius_to_parallax(radius)

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
        # TODO: Make function to cut the radius at some point, or add it to this function
        # TODO: Does center shift, xc, yc need to be applied?
        # TODO: Make tests for borders of inputs
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

            if (axis_ratio.value < 0.) or (axis_ratio.value > 1.):
                raise ValueError("The axis ratio must be between [0., 1.]")

            if (pos_angle.value < 0) or (pos_angle.value > 180):
                raise ValueError("The positional angle must be between [0, 180]")

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
            pos_angle = pos_angle.to(u.rad)

            # NOTE: This was taken from Jozsef's code and until here output is equal
            xr, yr = (x*np.cos(pos_angle)-y*np.sin(pos_angle))/axis_ratio,\
                (x*np.sin(pos_angle)+y*np.cos(pos_angle))
            radius = np.sqrt(xr**2+yr**2)
            self.axes_image, self.polar_angle = [xr, yr], np.arctan2(xr, yr)
        else:
            radius = np.sqrt(x**2+y**2)
            self.axes_image, self.polar_angle = [x, y], np.arctan2(x, y)

        return radius

    # TODO: Rework this function alike the others
    def _set_uv_grid(self, wavelength: float,
                     incline_params: List[float] = None,
                     uvcoords: np.ndarray = None,
                     vector: Optional[bool] = True) -> Quantity:
        """Sets the uv coords for visibility modelling

        Parameters
        ----------
        incline_params: List[float], optional
            A list of the three angles [axis_ratio, pos_angle, inc_angle]
        uvcoords: List[float], optional
            If uv-coords are given, then the visibilities are calculated for them
        vector: bool, optional
            Returns the baseline vector if toggled true, else the baselines

        Returns
        -------
        baselines: astropy.units.Quantity
            The baselines for the uvcoords [astropy.units.m]
        uvcoords: astropy.units.Quantity
            The axis used to calculate the baselines [astropy.units.m]
        """
        if not isinstance(wavelength, u.Quantity):
            wavelength *= u.um
        elif wavelength.unit != u.um:
            raise IOError("Enter the wavelength in [astropy.units.um] or unitless!")

        if uvcoords is None:
            axis = np.linspace(-self.image_size, size, sampling, endpoint=False)*u.m

            # Star overhead sin(theta_0)=1 position
            u, v = axis/wavelength.to(u.m),\
                axis[:, np.newaxis]/wavelength.to(u.m)

        else:
            axis = uvcoords/wavelength.to(u.m)
            u, v = np.array([uvcoord[0] for uvcoord in uvcoords]), \
                    np.array([uvcoord[1] for uvcoord in uvcoords])

        if angles is not None:
            try:
                if len(angles) == 2:
                    axis_ratio, pos_angle = incline_params
                else:
                    axis_ratio = incline_params[0]
                    pos_angle, inc_angle = map(lambda x: (x*u.deg).to(u.rad),
                                               incline_params[1:])

                u_inclined, v_inclined = u*np.cos(pos_angle)+v*np.sin(pos_angle),\
                        v*np.cos(pos_angle)-u*np.sin(pos_angle)

                if len(angles) > 2:
                    v_inclined = v_inclined*np.cos(inc_angle)

                baselines = np.sqrt(u_inclined**2+v_inclined**2)
                baseline_vector = baselines*wavelength.to(u.m)
                self.axes_complex_image = [u_inclined, v_inclined]
            except:
                raise IOError(f"{inspect.stack()[0][3]}(): Check input"
                              " arguments, ellipsis_angles must be of the form"
                              " either [pos_angle] or "
                              " [ellipsis_angle, pos_angle, inc_angle]")

        else:
            baselines = np.sqrt(u**2+v**2)
            baseline_vector = baselines*wavelength.to(u.m)
            self.axes_complex_image = [u, v]

        return baseline_vector if vector else baselines

    def _set_azimuthal_modulation(self, image: Quantity,
                                  modulation_angle: Quantity,
                                  amplitude: int = 1) -> Quantity:
        """Calculates the azimuthal modulation of the object

        Parameters
        ----------
        image: astropy.units.Quantity
            The model's image [astropy.units.Jy/px]
        polar_angle: astropy.units.Quantity
            The polar angle of the x, y-coordinates [astropy.units.rad]
        amplitude: int
            The 'c'-amplitude. Will be converted to [astropy.units.dimensionless_unscaled]

        Returns
        -------
        azimuthal_modulation: astropy.units.Quantity
            The azimuthal modulation [astropy.units.dimensionless_unscaled]
        """
        if not isinstance(modulation_angle, u.Quantity):
            modulation_angle *= u.deg
        elif modulation_angle.unit != u.deg:
            raise IOError("Enter the modulation angle in [astropy.units.deg] or unitless!")

        if not isinstance(amplitude, u.Quantity):
            amplitude *= u.dimensionless_unscaled
        elif amplitude.unit != u.dimensionless_unscaled:
            raise IOError("Enter the modulation angle in [astropy.units.deg] or unitless!")

        # TODO: Implement Modulation field like Jozsef?
        modulation_angle = modulation_angle.to(u.rad)
        total_mod = amplitude*np.cos(self.polar_angle-modulation_angle)
        image *= 1 + total_mod
        image.value[image.value < 0.] = 0.
        return image

    def _set_zeros(self, image: Quantity) -> Quantity:
        """Sets an image grid to all zeros"""
        return image*0

    def _set_ones(self, image: Quantity) -> Quantity:
        """Sets and image grid to all ones"""
        image[image != 0.] = 1.*image.unit
        return image

    def _temperature_gradient(self, radius: Quantity, power_law_exponent: float,
                              inner_radius: Optional[Quantity] = None,
                              inner_temperature: Optional[Quantity] = None
                              ) -> Quantity:
        """Calculates the temperature gradient

        Parameters
        ----------
        radius: astropy.units.Quantity
            An array containing all the points for the radius extending outwards
            [astropy.units.mas]
        power_law_exponent: float
            A float specifying the power law exponent of the temperature gradient "q"
        inner_radius: astropy.units.Quantity, optional
            The inner radius of the object, if not given then the sublimation radius is
            used [astropy.units.mas]
        inner_temperature: astropy.units.Quantity, optional

        Returns
        -------
        temperature_gradient: astropy.units.Quantity
            The temperature gradient [astropy.units.K]
        """
        if inner_radius is None:
            inner_radius = self.sublimation_radius
        elif not isinstance(inner_radius, u.Quantity):
            inner_radius *= u.mas
        elif inner_radius.unit != u.mas:
            raise IOError("Enter the inner radius in [astropy.units.mas] or unitless!")

        if inner_temperature is None:
            inner_temperature = self.sublimation_temperature
        elif not isinstance(inner_temperature, u.Quantity):
            inner_temperature *= u.K
        elif inner_temperature.unit != u.K:
            raise IOError("Enter the inner temperature in [astropy.units.K] or unitless!")

        return models.PowerLaw1D().evaluate(radius, inner_temperature,
                                            inner_radius, power_law_exponent)

    def _optical_depth_gradient(self, radius: Quantity,
                                inner_optical_depth: Quantity,
                                power_law_exponent: float,
                                inner_radius: Optional[Quantity] = None
                                ) -> Quantity:
        """Calculates the optical depth gradient

        Parameters
        ----------
        radius: astropy.units.Quantity
            An array containing all the points for the radius extending outwards
            [astropy.units.mas]
        inner_optical_depth: Quantity
            The optical depth at the inner radius [astropy.units.dimensionless_unscaled]
        power_law_exponent: float
            A float specifying the power law exponent of the temperature gradient "q"
        inner_radius: astropy.units.Quantity, optional
            The inner radius of the object, if not given then the sublimation radius is
            used [astropy.units.mas]

        Returns
        -------
        """
        if not isinstance(inner_optical_depth, u.Quantity):
            inner_optical_depth *= u.dimensionless_unscaled
        elif inner_optical_depth.unit != u.dimensionless_unscaled:
            raise IOError("Enter the inner optical depth in"\
                          " [astropy.units.dimensionless_unscaled] or unitless!")

        if inner_radius is None:
            inner_radius = self.sublimation_radius
        elif not isinstance(inner_radius, u.Quantity):
            inner_radius *= u.mas
        elif inner_radius.unit != u.mas:
            raise IOError("Enter the inner radius in [astropy.units.mas] or unitless!")

        return models.PowerLaw1D().evaluate(radius, inner_optical_depth,
                                            inner_radius, power_law_exponent)

    # TODO: Fix the tests here
    def _flux_per_pixel(self, wavelength: Quantity,
                       temperature_distribution: Quantity,
                       optical_depth: Quantity) -> Quantity:
        """Calculates the total flux of the model

        Parameters
        ----------
        wavelength: astropy.units.Quantity
            The wavelength to be used for the BlackBody calculation [astropy.units.um]
        temperature: astropy.units.Quantity
            The temperature distribution of the disc [astropy.units.K]
        optical_depth: astropy.units.Quantity
            The optical depth of the disc [astropy.units.dimensionless_unscaled]

        Returns
        -------
        flux: astropy.units.Quantity
            The object's flux per pixel [astropy.units.Jy]/px
        """
        if not isinstance(wavelength, u.Quantity):
            wavelength = (wavelength*u.um).to(u.AA)
        elif wavelength.unit == u.um:
            wavelength = wavelength.to(u.AA)
        else:
            raise IOError("Enter the wavelength in"\
                          " [astropy.units.um] or unitless!")

        if not isinstance(optical_depth, u.Quantity):
            temperature_distribution *= u.K
        elif temperature_distribution.unit != u.K:
            raise IOError("Enter the temperature distribution in"\
                          " [astropy.units.K] or unitless!")

        if not isinstance(optical_depth, u.Quantity):
            optical_depth *= u.dimensionless_unscaled
        elif optical_depth.unit != u.dimensionless_unscaled:
            raise IOError("Enter the optical depth distribution in"\
                          " [astropy.units.dimensionless_unscaled] or unitless!")

        # TODO: Check if the conversion of the Planck's law works from u.um to u.AA ->
        # Should be ok tho
        plancks_law = models.BlackBody(temperature=temperature_distribution)
        # NOTE: Convert sr to mas**2. Field of view = sr or mas**2
        spectral_radiance = plancks_law(wavelength).to(u.erg/(u.cm**2*u.Hz*u.s*u.mas**2))
        flux_per_pixel = spectral_radiance*self.field_of_view**2
        return flux_per_pixel.to(u.Jy)*(1-np.exp(-optical_depth))

    def _stellar_flux(self, wavelength: Quantity) -> Quantity:
        """Calculates the stellar flux from the distance and its radius

        Parameters
        ----------
        wavelength: astropy.units.Quantity
            The wavelength to be used for the BlackBody calculation [astropy.units.um]

        Returns
        -------
        stellar_flux: astropy.units.Quantity
            The star's flux [astropy.units.Jy]
        """
        if not isinstance(wavelength, u.Quantity):
            wavelength = (wavelength*u.um).to(u.AA)
        elif wavelength.unit == u.um:
            wavelength = wavelength.to(u.AA)
        else:
            raise IOError("Enter the wavelength in"\
                          " [astropy.units.um] or unitless!")

        plancks_law = models.BlackBody(temperature=self.effective_temperature)
        spectral_radiance = plancks_law(wavelength).to(u.erg/(u.cm**2*u.Hz*u.s*u.mas**2))
        stellar_radius = self._calculate_stellar_radius()
        # TODO: Check if that can be used in this context -> The conversion
        stellar_radius_angular = self._convert_orbital_radius_to_parallax(stellar_radius)
        return (spectral_radiance*np.pi*(stellar_radius_angular)**2).to(u.Jy)

    def eval_model(self, params: namedtuple) -> Quantity:
        """Evaluates the model image's radius

        Returns
        --------
        image: Quantity
            A two-dimensional model image [astropy.units.mas]
        """
        pass

    def eval_object(self, params: namedtuple) -> Quantity:
        """Evaluates the model image's object

        Returns
        --------
        image: Quantity
            A two-dimensional model image [astropy.units.dimensionless_unscaled]
        """
        return self._set_ones(self.eval_model(params)).value*u.dimensionless_unscaled

    def eval_vis(self, params: namedtuple) -> Quantity:
        """Evaluates the complex visibility function of the model.

        Returns
        -------
        complex_visibility_function: Quantity
            A two-dimensional complex visibility function [astropy.units.m]
        """
        pass

    # TODO: Properly implement this function at a later time
    def plot(self, image: Quantity) -> None:
        """Plots and image"""
        plt.imshow(image.value)
        plt.show()


if __name__ == "__main__":
    ...
