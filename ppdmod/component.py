from typing import Optional, Tuple

import astropy.units as u
import numpy as np
from astropy.modeling.models import BlackBody
from scipy.special import j0, jv

from ._spectral_cy import grid
from .fft import compute_real2Dfourier_transform
from .parameter import STANDARD_PARAMETERS, Parameter
from .options import OPTIONS
from .utils import rebin_image, upbin_image, get_new_dimension,\
        distance_to_angular, calculate_effective_baselines


class Component:
    """The base class for the component.

    Parameters
    ----------
    xx : float
        The x-coordinate of the component.
    yy : float
        The x-coordinate of the component.
    dim : float
        The dimension [px].
    """
    name = "Generic component"
    shortname = "GenComp"
    description = "This is the class from which all components are derived."
    _elliptic = False

    def __init__(self, **kwargs):
        """The class's constructor."""
        self.params = {}
        self.params["x"] = Parameter(**STANDARD_PARAMETERS["x"])
        self.params["y"] = Parameter(**STANDARD_PARAMETERS["y"])
        self.params["dim"] = Parameter(**STANDARD_PARAMETERS["dim"])
        self.params["pixel_size"] = Parameter(
            **STANDARD_PARAMETERS["pixel_size"])
        self.params["pa"] = Parameter(**STANDARD_PARAMETERS["pa"])
        self.params["elong"] = Parameter(**STANDARD_PARAMETERS["elong"])

        if not self.elliptic:
            self.params["pa"].free = False
            self.params["elong"].free = False
        self._eval(**kwargs)

    @property
    def elliptic(self) -> bool:
        """Gets if the component is elliptic."""
        return self._elliptic

    @elliptic.setter
    def elliptic(self, value: bool) -> None:
        """Sets the position angle and the parameters to free or false
        if elliptic is set."""
        if value:
            self.params["pa"].free = True
            self.params["elong"].free = True
        else:
            self.params["pa"].free = False
            self.params["elong"].free = False
        self._elliptic = value

    def _eval(self, **kwargs):
        """Sets the parameters (values) from the keyword arguments."""
        for key, value in kwargs.items():
            if key in self.params:
                if isinstance(value, Parameter):
                    self.params[key] = value
                else:
                    self.params[key].value = value

    def _calculate_internal_grid(
            self, dim: int, pixel_size: u.mas
            ) -> Tuple[u.Quantity[u.mas], u.Quantity[u.mas]]:
        """Calculates the model grid.

        Parameters
        ----------
        dim : float, optional
        pixel_size : float, optional

        Returns
        -------
        xx : astropy.units.mas
            The x-coordinate grid.
        yy : astropy.units.mas
            The y-coordinate grid.
        """
        elong, pa = self.params["elong"](), self.params["pa"]()
        elong = elong.value if elong is not None else elong
        pa = pa.value if pa is not None else pa
        return grid(dim, pixel_size.value, elong, pa, self.elliptic)

    def _translate_fourier_transform(self, ucoord: u.m, vcoord: u.m,
                                     wavelength: u.um) -> u.one:
        """Translate the coordinates of the fourier transform."""
        x, y = map(lambda x: self.params[x]().to(u.rad), ["x", "y"])
        ucoord, vcoord = map(
            lambda x: (u.Quantity(value=x, unit=u.m)/wavelength.to(u.m))/u.rad,
            [ucoord, vcoord])
        return np.exp(-2*1j*np.pi*(ucoord*x+vcoord*y)).value

    def _translate_coordinates(
            self, xx: u.mas, yy: u.mas
            ) -> Tuple[u.Quantity[u.mas], u.Quantity[u.mas]]:
        """Shifts the coordinates according to an offset."""
        xx, yy = map(lambda x: u.Quantity(value=x, unit=u.mas), [xx, yy])
        return xx-self.params["x"](), yy-self.params["y"]()


class AnalyticalComponent(Component):
    """Class for all analytically calculated components."""
    name = "Analytical Component"
    shortname = "AnaComp"
    description = "This is the class from which all"\
                  "analytical components are derived."

    def _image_function(self, xx: u.mas, yy: u.mas,
                        wavelength: Optional[u.Quantity[u.um]] = None
                        ) -> Optional[u.Quantity]:
        """Calculates the image from a 2D grid.

        Parameters
        ----------
        xx : astropy.units.mas
            The x-coordinate grid.
        yy : astropy.units.mas
            The y-coordinate grid.
        wavelength : u.m, optional

        Returns
        -------
        image : astropy.units.Quantity, Optional
        """
        return

    def _visibility_function(self, dim: int, pixel_size: u.mas,
                             wavelength: Optional[u.Quantity[u.um]] = None
                             ) -> np.ndarray:
        """Calculates the complex visibility of the the component's image.

        Parameters
        ----------
        wavelength : astropy.units.um, optional

        Returns
        -------
        complex_visibility_function : numpy.ndarray
        """
        return

    def calculate_image(self, dim: Optional[float] = None,
                        pixel_size: Optional[float] = None,
                        wavelength: Optional[u.Quantity[u.um]] = None) -> u.Jy:
        """Calculates a 2D image.

        Parameters
        ----------
        dim : float
            The dimension [px].
        pixel_size : float
            The size of a pixel [mas].
        wavelength : astropy.units.um, optional

        Returns
        -------
        image : astropy.units.Quantity
        """
        dim = self.params["dim"]() if dim is None else dim
        pixel_size = self.params["pixel_size"]()\
            if pixel_size is None else pixel_size
        dim = u.Quantity(value=dim, unit=u.one, dtype=int)
        pixel_size = u.Quantity(value=pixel_size, unit=u.mas)
        dim = get_new_dimension(
                dim, OPTIONS["fourier.binning"], OPTIONS["fourier.padding"])
        x_arr, y_arr = self._calculate_internal_grid(dim, pixel_size)
        return self._image_function(x_arr, y_arr, wavelength)

    def calculate_complex_visibility(
            self, dim: Optional[float] = None,
            pixel_size: Optional[float] = None,
            wavelength: Optional[u.Quantity[u.um]] = None) -> np.ndarray:
        """Calculates the complex visibility of the the component's image.

        Parameters
        ----------
        wavelength : astropy.units.um, optional

        Returns
        -------
        complex_visibility_function : numpy.ndarray
        """
        dim = self.params["dim"]() if dim is None else dim
        pixel_size = self.params["pixel_size"]()\
            if pixel_size is None else pixel_size
        dim = u.Quantity(value=dim, unit=u.one, dtype=int)
        pixel_size = u.Quantity(value=pixel_size, unit=u.mas)
        dim = get_new_dimension(
                dim, OPTIONS["fourier.binning"], OPTIONS["fourier.padding"])
        return self._visibility_function(dim, pixel_size, wavelength)


class HankelComponent(Component):
    """The base class for the component.

    Parameters
    ----------
    xx : float
        The x-coordinate of the component.
    yy : float
        The x-coordinate of the component.
    dim : float
        The dimension [px].
    """
    name = "Hankel Component"
    shortname = "HankComp"
    description = "This defines the analytical hankel transformation."
    elliptic = True
    asymmetric = False
    optically_thick = False
    const_temperature = False
    continuum_contribution = False

    def __init__(self, **kwargs):
        """The class's constructor."""
        super().__init__(**kwargs)
        self._stellar_angular_radius = None

        self.params["dist"] = Parameter(**STANDARD_PARAMETERS["dist"])
        self.params["eff_temp"] = Parameter(**STANDARD_PARAMETERS["eff_temp"])
        self.params["eff_radius"] = Parameter(**STANDARD_PARAMETERS["eff_radius"])

        self.params["rin0"] = Parameter(**STANDARD_PARAMETERS["rin0"])

        self.params["rin"] = Parameter(**STANDARD_PARAMETERS["rin"])
        self.params["rout"] = Parameter(**STANDARD_PARAMETERS["rout"])

        self.params["a"] = Parameter(**STANDARD_PARAMETERS["a"])
        self.params["phi"] = Parameter(**STANDARD_PARAMETERS["phi"])

        self.params["q"] = Parameter(**STANDARD_PARAMETERS["q"])
        self.params["inner_temp"] = Parameter(**STANDARD_PARAMETERS["inner_temp"])

        self.params["p"] = Parameter(**STANDARD_PARAMETERS["p"])
        self.params["inner_sigma"] = Parameter(**STANDARD_PARAMETERS["inner_sigma"])
        self.params["kappa_abs"] = Parameter(**STANDARD_PARAMETERS["kappa_abs"])
        self.params["cont_weight"] = Parameter(**STANDARD_PARAMETERS["cont_weight"])
        self.params["kappa_cont"] = Parameter(**STANDARD_PARAMETERS["kappa_cont"])

        if self.const_temperature:
            self.params["q"].free = False
            self.params["inner_temp"].free = False

        if not self.asymmetric:
            self.params["a"].free = False
            self.params["phi"].free = False

        if not self.continuum_contribution:
            self.params["cont_weight"].free = False
        self._eval(**kwargs)

    @property
    def stellar_radius_angular(self) -> u.mas:
        r"""Calculates the parallax from the stellar radius and the distance to
        the object.

        Returns
        -------
        stellar_radius_angular : astropy.units.mas
            The parallax of the stellar radius.
        """
        self._stellar_angular_radius = distance_to_angular(
            self.params["eff_radius"](), self.params["dist"]())
        return self._stellar_angular_radius

    def _calculate_internal_grid(self, dim: int) -> u.mas:
        """Calculates the model grid.

        Parameters
        ----------
        dim : float

        Returns
        -------
        radial_grid : astropy.units.mas
            A one dimensional linear or logarithmic grid.
        """
        rin, rout = self.params["rin"](), self.params["rout"]()
        if OPTIONS["model.gridtype"] == "linear":
            radius = np.linspace(rin.value, rout.value, dim)*self.params["rin"].unit
        else:
            radius = np.logspace(np.log10(rin.value),
                               np.log10(rout.value), dim)*self.params["rin"].unit
        return radius.astype(OPTIONS["model.dtype.real"])
    

    def _get_opacity(self, wavelength: u.um) -> u.cm**2/u.g:
        """Set the opacity from wavelength."""
        if self.continuum_contribution:
            opacity = self.params["kappa_abs"](wavelength) +\
                      self.params["cont_weight"]() *\
                      self.params["kappa_cont"](wavelength)
        else:
            opacity = self.params["kappa_abs"](wavelength)
        return opacity.astype(OPTIONS["model.dtype.real"])

    def _azimuthal_modulation(self, xx: u.mas, yy: u.mas) -> u.one:
        """Calculates the azimuthal modulation."""
        if not self.asymmetric:
            return 1
        azimuthal_modulation = (1+self.params["a"]()\
                * np.cos(np.arctan2(yy, xx)-self.params["phi"]()))
        return azimuthal_modulation.astype(OPTIONS["model.dtype.real"])

    def _temperature_profile_function(
            self, radius: u.mas, innermost_radius: u.mas) -> u.K:
        """Calculates a 1D-temperature profile."""
        if self.const_temperature:
            temperature_profile = np.sqrt(self.stellar_radius_angular/(2.0*radius))\
                    * self.params["eff_temp"]()
        else:
            temperature_profile = self.params["inner_temp"]()\
                    * (radius/innermost_radius)**(-self.params["q"]())
        return temperature_profile.astype(OPTIONS["model.dtype.real"])

    def _surface_density_profile_function(
            self, radius: u.mas, innermost_radius: u.mas) -> u.one:
        """Calculates a 1D-surface density profile."""
        surface_density = self.params["inner_sigma"]()\
                * (radius/innermost_radius)**(-self.params["p"]())
        return surface_density.astype(OPTIONS["model.dtype.real"])

    def _thickness_profile_function(
            self, radius: u.mas, innermost_radius: u.mas, wavelength: u.um) -> u.one:
        """Calculates a 1D-thickness profile."""
        if wavelength.shape == ():
            wavelength.reshape((wavelength.size,))
        if self.optically_thick:
            return 1
        surface_density_profile = self._surface_density_profile_function(
            radius, innermost_radius)
        thickness_profile = (1-np.exp(-surface_density_profile*self._get_opacity(wavelength)))
        return thickness_profile.astype(OPTIONS["model.dtype.real"])

    def _brightness_profile_function(self, radius: u.mas, wavelength: u.um) -> u.Jy:
        """Calculates a 1D-brightness profile from a dust-surface density- and
        temperature profile.

        Parameters
        ----------
        wl : astropy.units.um
            Wavelengths.

        Returns
        -------
        brightness_profile : astropy.units.Jy
        """
        if wavelength.shape == ():
            wavelength.reshape((wavelength.size,))
        innermost_radius = self.params["rin0"]()\
            if self.params["rin0"]() != 0 else self.params["rin"]()

        # TODO: Think of a way how to implement the innermost radius
        # and at the same time keep the grid as it is.
        temperature_profile = self._temperature_profile_function(
                radius, innermost_radius)
        brightness_profile = BlackBody(temperature_profile)(wavelength)
        thickness_profile = self._thickness_profile_function(
                radius, innermost_radius, wavelength)
        brightness_profile *= thickness_profile*OPTIONS["model.flux.factor"]
        return brightness_profile.astype(OPTIONS["model.dtype.real"])

    def calculate_image(self, dim: int, pixel_size: u.mas, wavelength: u.um) -> u.Jy:
        """Calculates the image."""
        pixel_size = pixel_size if isinstance(pixel_size, u.Quantity) else pixel_size*u.mas
        xx = np.linspace(-0.5, 0.5, dim)*dim*pixel_size
        xx, yy = np.meshgrid(xx, xx)
        if self.elliptic:
            pa, elong = self.params["pa"](), self.params["elong"]()
            xx = xx*np.cos(pa)-yy*np.sin(pa)
            yy = (xx*np.sin(pa)+yy*np.cos(pa))/elong
        azimuthal_modulation = self._azimuthal_modulation(xx, yy)
        radius = np.hypot(xx, yy)
        
        radial_profile = np.logical_and(radius >= self.params["rin"](),
                                        radius <= self.params["rout"]())
        brightness_profile = self._brightness_profile_function(
                radius, wavelength).to(u.erg/(u.cm**2*u.rad**2*u.s*u.Hz))\
                        * pixel_size.to(u.rad)**2
        image = brightness_profile.to(u.Jy)*radial_profile*azimuthal_modulation
        return image.astype(OPTIONS["model.dtype.real"])

    def hankel_transform(self, brightness_profile: u.erg/(u.rad**2*u.s*u.Hz),
                         radius: u.mas, ucoord: u.m,
                         vcoord: u.m, wavelength: u.um,
                         star_flux: Optional[u.Quantity[u.Jy]] = None,
                         total_flux: Optional[u.Quantity[u.Jy]] = None) -> np.ndarray:
        """Executes the hankel transform."""
        # pad = 1 if OPTIONS['FTPaddingFactor'] is None else OPTIONS['FTPaddingFactor']
        # fov = radius[-1]-radius[0]
        # dsfreq0 = 1/(fov*pad).value
        # sfreq0 = np.linspace(0, pad*nr-1, pad*nr)*dsfreq0
        star_flux = 0*u.Jy if star_flux is None else star_flux
        total_flux = 1*u.Jy if total_flux is None else total_flux
        radius = radius.to(u.rad)
        baseline_groups, baseline_angle_groups = calculate_effective_baselines(
                ucoord, vcoord, self.params["elong"](), self.params["pa"]())
        baseline_groups /= wavelength.to(u.m).value*u.rad

        if len(baseline_groups.shape) == 1:
            baseline_groups = baseline_groups[np.newaxis, :]
            baseline_angle_groups = baseline_angle_groups[np.newaxis, :]

        visibilities, modulations = [], [[] for _ in range(1, OPTIONS["model.modulation.order"]+1)]
        for baselines, baseline_angles in zip(baseline_groups, baseline_angle_groups):
            for baseline, baseline_angle in zip(baselines, baseline_angles):
                visibility = 2*np.pi*np.trapz(
                        radius*brightness_profile*j0(2.*np.pi*radius.value*baseline.value), radius)
                visibilities.append(visibility.to(u.Jy))

                # TODO: Think of a way to implement more parameters for the azimuthal modulation.
                if self.asymmetric:
                    for order in range(1, OPTIONS["model.modulation.order"]+1):
                        modulation = (-1j)**order*self.params["a"]()\
                                * np.cos(order*(baseline_angle-self.params["phi"]().to(u.rad)))\
                                * np.trapz(radius*brightness_profile\
                                * jv(order, 2.*np.pi*radius.value*baseline.value), radius)
                        modulations[order-1].append(modulation.to(u.Jy))
        visibilities = (u.Quantity(visibilities, unit=u.Jy, dtype=np.complex64)+star_flux)/total_flux
        modulations = u.Quantity(modulations, unit=u.Jy, dtype=np.complex64)/total_flux
        return visibilities.astype(OPTIONS["model.dtype.complex"]),\
                modulations.astype(OPTIONS["model.dtype.complex"])

    def calculate_total_flux(self, wavelength: u.um,
                             star_flux: Optional[u.Quantity[u.Jy]] = None) -> u.Jy:
        """Calculates the total flux from the hankel transformation."""
        star_flux = 0*u.Jy if star_flux is None else star_flux
        radius = self._calculate_internal_grid(self.params["dim"]())
        brightness_profile = self._brightness_profile_function(radius, wavelength)
        total_flux = (2.*np.pi*np.trapz(radius*brightness_profile, radius).to(u.Jy)
                      + star_flux).value
        return np.abs(total_flux.astype(OPTIONS["model.dtype.real"]))

    def calculate_visibility(self, ucoord: u.m, vcoord: u.m,
                             wavelength: u.um, **kwargs) -> np.ndarray:
        """Calculates the visibilities via hankel transformation."""
        radius = self._calculate_internal_grid(self.params["dim"]())
        vis, vis_mod = self.hankel_transform(
                self._brightness_profile_function(radius, wavelength),
                radius, ucoord, vcoord, wavelength, **kwargs)
        if vis_mod.size != 0:
            vis += vis_mod.sum(0)
        return np.abs(vis.value).astype(OPTIONS["model.dtype.real"])

    def calculate_closure_phase(self, ucoord: u.m, vcoord: u.m,
                                wavelength: u.um, **kwargs) -> np.ndarray:
        """Calculates the closure phases via hankel transformation."""
        radius = self._calculate_internal_grid(self.params["dim"]())
        vis, vis_mod = self.hankel_transform(
                self._brightness_profile_function(radius, wavelength),
                radius, ucoord, vcoord, wavelength, **kwargs)
        vis = vis.reshape(ucoord.shape)
        vis = np.vstack((vis[:2], vis[-1].conj()))
        if vis_mod.size != 0:
            for mod in vis_mod:
                mod = mod.reshape(ucoord.shape)
                vis += np.vstack((mod[:2], mod[-1].conj()))
        return np.angle(np.prod(vis.value, axis=0),
                        deg=True).astype(OPTIONS["model.dtype.real"])


class NumericalComponent(Component):
    """Base class with increased computational performance for numerical
    calculations.

    Parameters
    ----------
    pixel_size : float
        The size of a pixel [mas].
    pa : float
        Positional angle [deg].
    elong : float
        Elongation of the disk [dimensionless].
    dim : float
        The dimension [px].

    Attributes
    ----------

    Notes
    -----
    This class will automaticall set-up a cache directory in order to speed up
    the calculations. The directory itself can be set via the 'cache_dir' keyword.
    """
    elliptic = False

    def _image_function(self, xx: u.mas,
                        yy: u.mas, wavelength: u.um) -> u.Jy:
        """Calculates the image from a 2D grid.

        Parameters
        ----------
        xx : astropy.units.mas
            The x-coordinate grid.
        yy : astropy.units.mas
            The y-coordinate grid.
        wavelength : astropy.units.um, optional

        Returns
        -------
        image : astropy.units.Quantity, optional
        """
        return

    def calculate_image(self, dim: Optional[float] = None,
                        pixel_size: Optional[float] = None,
                        wavelength: Optional[u.Quantity[u.um]] = None) -> u.Jy:
        """Calculates a 2D image.

        Parameters
        ----------
        dim : float
            The dimension [px].
        pixel_size : float
            The size of a pixel [mas].
        wavelength : astropy.units.um, optional

        Returns
        -------
        image : astropy.units.Quantity
        """
        dim = self.params["dim"]() if dim is None else dim
        pixel_size = self.params["pixel_size"]()\
            if pixel_size is None else pixel_size
        dim = u.Quantity(value=dim, unit=u.one, dtype=int)
        pixel_size = u.Quantity(value=pixel_size, unit=u.mas)

        if OPTIONS["model.matryoshka"]:
            image = None
            new_dim = get_new_dimension(
                    dim, OPTIONS["model.matryoshka.binning_factors"][0])
            for binning_factor in OPTIONS["model.matryoshka.binning_factors"]:
                binning_factor = binning_factor if binning_factor is not None else 0
                if image is None:
                    x_arr, y_arr = self._calculate_internal_grid(
                            new_dim, pixel_size*2**binning_factor)
                    image_part = self._image_function(x_arr, y_arr, wavelength)
                    image = upbin_image(image_part, binning_factor)\
                            * (2**binning_factor*2**binning_factor)
                else:
                    x_arr, y_arr = self._calculate_internal_grid(
                            new_dim, pixel_size*2**-binning_factor)
                    image_part = self._image_function(x_arr, y_arr, wavelength)
                    image_part = rebin_image(image_part, binning_factor)
                    start = (image.shape[0]-image_part.shape[0])//2
                    end = start + image_part.shape[0]
                    image[start:end, start:end] =\
                            image_part/(2**binning_factor*2**binning_factor)
        else:
            x_arr, y_arr = self._calculate_internal_grid(dim, pixel_size)
            image = self._image_function(x_arr, y_arr, wavelength)
        if OPTIONS["fourier.binning"] is not None:
            image = rebin_image(image, OPTIONS["fourier.binning"])
        if OPTIONS["fourier.padding"] is not None:
            image = pad_image(image, OPTIONS["fourier.padding"])
        return image

    def calculate_complex_visibility(
            self, dim: Optional[float] = None,
            pixel_size: Optional[float] = None,
            wavelength: Optional[u.Quantity[u.um]] = None) -> np.ndarray:
        """Calculates the complex visibility of the the component's image.

        Parameters
        ----------
        wavelength : astropy.units.um, optional

        Returns
        -------
        complex_visibility_function : numpy.ndarray
        """
        dim = self.params["dim"]() if dim is None else dim
        pixel_size = self.params["pixel_size"]()\
            if pixel_size is None else pixel_size
        dim = u.Quantity(value=dim, unit=u.one, dtype=int)
        pixel_size = u.Quantity(value=pixel_size, unit=u.mas)
        image = self.calculate_image(dim, pixel_size, wavelength)
        return compute_real2Dfourier_transform(image.value)
