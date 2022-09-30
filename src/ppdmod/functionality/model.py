import inspect
import numpy as np
import astropy.units as u
import astropy.constants as c

from astropy.modeling import models
from astropy.units import Quantity
from typing import List, Union, Optional

# TODO: Implement FFT as a part of the base_model_class, maybe?
# TODO: Think about calling FFT class in model class to evaluate model
# TODO: Add checks for right unit input in all the files


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
    def __init__(self, field_of_view: Quantity, image_size: int,
                 sublimation_temperature: int, effective_temperature: int,
                 distance: int, luminosity_star: int,
                 pixel_sampling: Optional[int] = None) -> None:
        # TODO: Maybe make a specific save name for the model also
        self.name = None
        self.axes_image, self.axes_complex_image, self.polar_angle = None, None, None

        self._field_of_view = field_of_view*u.mas
        self._image_size = image_size*u.dimensionless_unscaled
        self._pixel_sampling = self.image_size if pixel_sampling\
            is None else pixel_sampling
        self._sublimation_temperature = sublimation_temperature*u.K
        self._effective_temperature = effective_temperature*u.K
        self._luminosity_star = luminosity_star*c.L_sun
        self._distance = distance*u.pc

        # self._stellar_radius = stellar_radius_pc(self.T_eff, self.L_star)
        # self.stellar_flux = np.pi*(self._stellar_radius/self.d)**2*\
                # self._stellar_radians*1e26

    @property
    def field_of_view(self):
        return self._field_of_view

    @field_of_view.setter
    def field_of_view(self, value):
        self._field_of_view = value*u.mas

    @property
    def image_size(self):
        return self._image_size

    @image_size.setter
    def image_size(self, value):
        self._image_size = value*u.mas

    @property
    def pixel_sampling(self):
        return self._pixel_sampling

    @pixel_sampling.setter
    def pixel_sampling(self, value):
        self._pixel_sampling = value*u.mas

    @property
    def pixel_scaling(self):
        return self.field_of_view/self.pixel_sampling

    @property
    def sublimation_temperature(self):
        return self._field_of_view

    @sublimation_temperature.setter
    def sublimation_temperature(self, value):
        if not isinstace(value, u.Quantity):
            value *= u.K
        self._sublimation_temperature = value

    @property
    def effective_temperature(self):
        return self._effective_temperature

    @effective_temperature.setter
    def effective_temperature(self, value):
        self._effective_temperature = value*u.K

    @property
    def luminosity_star(self):
        return self._luminosity_star

    @luminosity_star.setter
    def luminosity_star(self, value):
        self._luminosity_star = value*c.L_sun

    @property
    def distance(self):
        return self._distance

    @distance.setter
    def distance(self, value):
        self._distance = value*u.pc

    # TODO: Make repr function that tells what the model has been initialised with
    def _calculate_orbital_radius_from_parallax(self, parallax: Quantity) -> Quantity:
        """Calculates the orbital radius from a parallax [astropy.units.arcsec] and gives
        it in [astropy.units.m]

        Parameters
        ----------
        parallax: astropy.units.Quantity
            The angle of the orbital radius [astropy.units.arcsec]

        Returns
        -------
        orbital_radius: astropy.units.Quantity
            The orbital radius [astropy.units.m]
        """
        return (parallax.to('', equivalencies=u.dimensionless_angles())\
                *self.distance).to(u.m)

    def _calculate_stellar_radius(self) -> Quantity:
        """Calculates the stellar radius from its attributes and converts it from
        m to parsec

        Returns
        -------
        stellar_radius: float
            The star's radius [astropy.units.pc]
        """
        return (np.sqrt(self.luminosity_star/\
                       (4*np.pi*c.sigma_sb*self.effective_temperature**4))).to(u.pc)

    def _calculate_stellar_radians(self, wavelength: Quantity) -> Quantity:
        """Calculates the flux from the central star

        Parameters
        ----------
        wavelength: astropy.units.Quantity
            The wavelength to be used for the BlackBody calculation [astropy.units.um]

        Returns
        -------
        stellar_radians: astropy.units.Quantity
            The flux of the star for a specific wavelength [astropy.units.Jy/px]
        """
        stellar_radians = models.BlackBody(temperature=self.effective_temperature)
        return stellar_radians(wavelength)

    def _calculate_sublimation_temperature(self,
                                           inner_radius: Optional[Quantity] = None
                                           ) -> Quantity:
        """Calculates the sublimation temperature at the inner rim of the disc

        Parameters
        ----------
        inner_radius: astropy.units.Quantity, optional
            The inner radius of the disc [astropy.units.mas]

        Returns
        -------
        sublimation_temperature: astropy.units.Quantity
            The sublimation temperature [astropy.units.K]
        """
        if inner_radius is None:
            inner_radius = self.sublimation_radius
        inner_radius = inner_radius.to(u.arcsec)
        return (self.luminosity_star/(4*np.pi*c.k_B*inner_radius**2))**(1/4)

    def _calculate_sublimation_radius(self) -> Quantity:
        """Calculates the sublimation radius at the inner rim of the disc

        Returns
        -------
        sublimation_radius: astropy.units.Quantity
            The sublimation radius [astropy.units.mas]
        """
        sublimation_radius = np.sqrt(self.luminosity_star/\
                                     (4*np.pi*c.sigma_sb*self.sublimation_temperature**4))
        return m2mas(sub_radius_m, self.distance)

    def _calculate_temperature_gradient(self, radius: Quantity, power_law_exponent: float,
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
        elif inner_temperature is None:
            inner_temperature = self.sublimation_temperature
        else:
            raise IOError("Either the inner radius or the inner temperature can be"\
                          "fixed, not both!")

        return models.PowerLaw1D().evaluate(radius, inner_temperature,
                                            inner_radius, power_law_exponent)

    def _calculate_optical_depth_gradient(self, radius: Quantity,
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
        if inner_radius is None:
            inner_radius = self.sublimation_radius
        return models.PowerLaw1D().evaluate(radius, inner_optical_depth,
                                            inner_radius, power_law_exponent)

    def _calculate_flux_per_pixel(self) -> Quantity:
        """Calculates the total flux of the model

        Parameters
        ----------

        Returns
        -------
        flux: astropy.units.Quantity
            The object's flux per pixel [astropy.units.Jy/px]
        """
        flux = blackbody(wavelength)
        flux *= (1-np.exp(-tau))*sr2mas(self._mas_size, self._sampling)
        flux[np.where(np.isnan(flux))],flux[np.where(np.isinf(flux))] = 0., 0.
        return flux*1e26

    def _calculate_azimuthal_modulation(self, image: Quantity,
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
        # TODO: Implement Modulation field like Jozsef?
        modulation_angle = (modulation_angle*u.deg).to(u.rad)
        total_mod = (amplitude*u.dimensionless_unscaled*\
                     np.cos(self.polar_angle-modulation_angle))
        image *= 1 + total_mod
        image.value[image.value < 0.] = 0.
        return image

    def _calculate_total_flux(self, *args) -> Quantity:
        """Sums up the flux from the individual pixel [astropy.units.Jy/px] brightness
        distribution to the complete brightness [astropy.units.Jy]"""
        return np.sum(self.get_flux(*args))

    def set_grid(self, incline_params: Optional[List[float]] = None) -> Quantity:
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
        # Make function to cut the radius at some point, or add it to this function
        x = np.linspace(-self.image_size//2, self.image_size//2,
                        self.pixel_sampling, endpoint=False)*self.pixel_scaling
        y = x[:, np.newaxis]

        if incline_params:
            try:
                axis_ratio, pos_angle = incline_params
                pos_angle = (pos_angle*u.deg).to(u.rad)
            except:
                raise IOError(f"{inspect.stack()[0][3]}(): Check input"
                              " arguments, 'incline_params' must be of the"
                              " form [axis_ratio, pos_angle]")

            if axis_ratio < 1.:
                raise ValueError("The axis_ratio has to be bigger than 1.")

            if (pos_angle > 0) and (pos_angle < 180):
                raise ValueError("The positional angle must be between [0, 180]")

            axis_ratio *= u.dimensionless_unscaled
            pos_angle = (pos_angle*u.deg).to(u.rad)

            xr, yr = x*np.cos(pos_angle)+y*np.sin(pos_angle),\
                    (-x*np.sin(pos_angle)+y*np.cos(pos_angle))/axis_ratio
            radius = np.sqrt(xr**2+yr**2)
            self.axes_image, self.polar_angle = [xr, yr], np.arctan2(xr, yr)
        else:
            radius = np.sqrt(x**2+y**2)
            self.axes_image, self.polar_angle = [x, y], np.arctan2(x, y)

        return radius

    def set_uv_grid(self, wavelength: float,
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
        # TODO: Work to split this from image_size -> (u, v)-coords should be separate
        wavelength *= u.um
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

    def eval_model(self) -> Quantity:
        """Evaluates the model image

        Returns
        --------
        image: Quantity
            A two-dimensional model image [astropy.units.mas]
        """
        pass

    def eval_vis(self) -> Quantity:
        """Evaluates the complex visibility function of the model.

        Returns
        -------
        complex_visibility_function: Quantity
            A two-dimensional complex visibility function [astropy.units.m]
        """
        pass


# TODO: Make this class combine individual models
class CombinedModel:
    # TODO: Think of how to combine the models
    def __init__(self):
        ...

if __name__ == "__main__":
    model = Model(50, 128, 1500, 7900, 1, 19)
    print(model._calculate_orbital_radius_from_parallax(1*u.arcsec))

