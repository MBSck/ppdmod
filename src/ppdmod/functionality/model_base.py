import inspect
import numpy as np
import astropy.units as u
import astropy.constants as c

from astropy.modeling import models
from typing import Any, List, Union, Optional, Callable

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
                 T_sub: int, T_eff: int, L_star: int, distance: int,
                 wavelength: float, pixel_sampling: Optional[int] = None) -> None:
        # TODO: Maybe make a specific save name for the model also
        self.name = None
        self.wl = wavelength*u.um
        self.axes_image, self.polar_angle = None, None
        self.image_size = image_size*u.dimensionless_unscaled
        self.field_of_view = field_of_view*u.mas
        self.pixel_sampling = self.image_size if self.pixel_sampling\
            is None else self.pixel_sampling

        self.T_sub, self.T_eff, self.L_star, self.d, self.wl = T_sub, T_eff, \
                L_star, self.distance, self.wl

        self._r_sub = sublimation_radius(self.T_sub, self.L_star, self.d)
        self._stellar_radius = stellar_radius_pc(self.T_eff, self.L_star)
        self._stellar_radians = plancks_law_nu(self.T_eff, self.wl)
        self.stellar_flux = np.pi*(self._stellar_radius/self.d)**2*\
                self._stellar_radians*1e26

    @property
    def pixel_scaling(self):
        """Calculates the pixel scale [astropy.units.mas/px]"""
        return (self.mas_size*u.mas)/self.pixel_sampling

    @property
    def r_sub(self):
        """Calculates the sublimation radius"""
        return self._r_sub

    @r_sub.setter
    def r_sub(self, value):
        """Sets the sublimation radius"""
        self._r_sub = value

    def get_total_flux(self, *args) -> np.ndarray:
        """Sums up the flux from [Jy/px] to [Jy]"""
        return np.sum(self.get_flux(*args))

    def _stellar_radius_pc(self, T_eff: int, L_star: int):
        """Calculates the stellar radius from its attributes and converts it from
        m to parsec

        Parameters
        ----------
        T_eff: int
            The star's effective temperature [K]
        L_star: int
            The star's luminosity [L_sun]

        Returns
        -------
        stellar_radius: float
            The star's radius [pc]
        """
        stellar_radius_m = np.sqrt((L_star*c.L_sun)/(4*np.pi*c.sigma_sb*T_eff**4))
        return stellar_radius_m/PARSEC2M

    def _sublimation_temperature(self, r_sub: float, L_star: int, distance: int):
        """Calculates the sublimation temperature at the inner rim of the disk

        Parameters
        ----------
        r_sub: float
            The sublimation radius [mas]
        L_star: int
            The star's luminosity in units of nominal solar luminosity
        distance: int
            Distance in parsec

        Returns
        -------
        T_sub: float
            The sublimation temperature [K]
        """
        r_sub /= m2mas(1, distance)
        return ((L_star*c.L_sun)/(4*np.pi*c.k_B*r_sub**2))**(1/4)

    def _sublimation_radius(self, T_sub: int, L_star: int, distance: int):
        """Calculates the sublimation radius of the disk

        Parameters
        ----------
        T_sub: int
            The sublimation temperature of the disk. Usually fixed to 1500 K
        L_star: int
            The star's luminosity in units of nominal solar luminosity
        distance: int
            Distance in parsec

        Returns
        -------
        R_sub: int
            The sublimation_radius [mas]
        """
        sub_radius_m = np.sqrt((L_star*c.L_sun)/(4*np.pi*c.sigma_sb*T_sub**4))
        return m2mas(sub_radius_m, distance)

    def _get_flux(self, tau_0: float, q: float, p: float,
                 r_sub: Optional[Union[int, float]] = None) -> np.array:
        """Calculates the total flux of the model

        Parameters
        ----------
        tau_0: float
            The optical depth of the disk, value between 0-1, which 1 being
            a perfect black body
        q: float
            The power law exponent of temperature
        p: float
            The power law exponent of optical depth
        r_sub: int | float, optional
            The inner radius used to calculate the inner/sublimation
            temperature, if provided

        Returns
        -------
        flux: np.ndarray
        """
        with np.errstate(divide='ignore'):
            if r_sub is not None:
                sub_temperature = sublimation_temperature(r_sub, self.L_star, self.d)
            else:
                sub_temperature = self._r_sub

            temperature = models.PowerLaw1D(self._radius, self._r_sub, q, self.T_sub)
            tau = models.PowerLaw1D(self._radius, self._r_sub, p, tau_0)
            blackbody = models.BlackBody(temperature=temperature*u.K)

            flux = blackbody(self.wl)
            flux *= (1-np.exp(-tau))*sr2mas(self._mas_size, self._sampling)
            flux[np.where(np.isnan(flux))],flux[np.where(np.isinf(flux))] = 0., 0.
            return flux*1e26


    def azimuthal_modulation(self, image, modulation_angle: float,
                             amplitude: int  = 1) -> Quantity:
        """Azimuthal modulation of an object

        Parameters
        ----------
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
        # TODO: Implement check that checks is polar angle is rad if not convert it
        modulation_angle = (modulation_angle*u.deg).to(u.rad)
        total_mod = (amplitude*u.dimensionless_unscaled*\
                     np.cos(self.polar_angle-modulation_angle))
        image *= 1 + total_mod
        image.value[image.value < 0.] = 0.
        return image

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
        x = np.linspace(-self.image_size//2, self.image_size//2,
                        self.pixel_sampling, endpoint=False)*self.pixel_scaling
        y = x[:, np.newaxis]

        if incline_params:
            try:
                axis_ratio, pos_angle = incline_params
                pos_angle = (pos_angle*u.deg).to(u.rad)
            except:
                raise IOError(f"{inspect.stack()[0][3]}(): Check input"
                              " arguments, ellipsis_angles must be of the"
                              " form [axis_ratio, pos_angle]")

            if axis_ratio < 1.:
                raise ValueError("The axis_ratio has to be bigger than 1.")

            if (pos_angle < 0) or (pos_angle > 180):
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

    def set_uv_grid(self, incline_params: List[float] = None,
                    uvcoords: np.ndarray = None, B: Optional[bool] = True) -> Quantity:
        """Sets the uv coords for visibility modelling

        Parameters
        ----------
        wavelength: float
            The wavelength the (u,v)-plane is sampled at
        sampling: int
            The pixel sampling
        size: int, optional
            Sets the range of the (u,v)-plane in meters, with size being the
            longest baseline
        angles: List[float], optional
            A list of the three angles [ellipsis_angle, pos_angle inc_angle]
        uvcoords: List[float], optional
            If uv-coords are given, then the visibilities are calculated for
        B: bool, optional
            Returns the baseline vector if toggled true, else the r vector

        Returns
        -------
        baselines: ArrayLike
            The baselines for the uvcoords
        uvcoords: ArrayLike
            The axis used to calculate the baselines
        """
        # TODO: Work to split this from image_size -> (u, v)-coords should be separate
        if uvcoords is None:
            axis = np.linspace(-self.image_size, size, sampling, endpoint=False)

            # Star overhead sin(theta_0)=1 position
            u, v = axis/wavelength, axis[:, np.newaxis]/wavelength

        else:
            axis = uvcoords/wavelength
            u, v = np.array([i[0] for i in uvcoords]), \
                    np.array([i[1] for i in uvcoords])

        if angles is not None:
            try:
                if len(angles) == 1:
                    axis_ratio, pos_angle = incline_params
                    ur, vr = u*np.cos(pos_angle)+v*np.sin(pos_angle), \
                            v*np.cos(pos_angle)-u*np.sin(pos_angle)
                    baselines = np.sqrt(ur**2+vr**2)
                    baseline_vector = baselines*self.wl.to(u.m) if B else baselines
                else:
                    axis_ratio = incline_params[0]
                    pos_angle, inc_angle = map(lambda x: (x*u.deg).to(u.rad),
                                               incline_params[1:])

                    ur, vr = u*np.cos(pos_angle)+v*np.sin(pos_angle), \
                            (v*np.cos(pos_angle)-u*np.sin(pos_angle))/axis_ratio
                    baselines = np.sqrt(ur**2+vr**2*np.cos(inc_angle)**2)
                    baseline_vector = baselines*self.wl.to(u.m) if B else baselines

                self.axes_complex_image = [ur, vr]
            except:
                raise IOError(f"{inspect.stack()[0][3]}(): Check input"
                              " arguments, ellipsis_angles must be of the form"
                              " either [pos_angle] or "
                              " [ellipsis_angle, pos_angle, inc_angle]")

        else:
            baselines = np.sqrt(u**2+v**2)
            baseline_vector = baselines*self.wl.to(u.m) if B else baselines
            self.axes_complex_image = [u, v]

        return baseline_vector

    def eval_model() -> Quantity:
        """Evaluates the model image
        Convention to put non fitting parameters at end of *args.

        Returns
        --------
        image: Quantity
            A two-dimensional model image [astropy.units.mas]
        """
        pass

    def eval_vis() -> Quantity:
        """Evaluates the visibilities of the model.
        Convention to put non fitting parameters at end of *args.

        Returns
        -------
        complex_image: Quantity
        """
        pass


if __name__ == "__main__":
    model = Model(1500, 7900, 140, 8)

