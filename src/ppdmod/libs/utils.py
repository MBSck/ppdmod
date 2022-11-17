import numpy as np
import astropy.units as u
import astropy.constants as c

from astropy.modeling import models
from types import SimpleNamespace
from typing import List, Tuple, Union, Optional
from astropy.units import Quantity


################################ PHYSICS #################################################

def calculate_effective_baselines(uv_coords: u.m,
                                  axis_ratio: u.dimensionless_unscaled,
                                  pos_angle: u.rad,
                                  wavelength: u.um) -> u.dimensionless_unscaled:
    """"""
    u_coords, v_coords = map(lambda x: x.squeeze(), np.split(uv_coords, 2, axis=1))
    projected_baselines = np.sqrt(u_coords**2+v_coords**2)
    projected_baselines_a_rad = np.arctan2(u_coords, v_coords)
    atd = np.arctan2(np.sin(pbla_rad-PA),(np.cos(pbla_rad-PA)))
    ucoords_eff = pbl*(np.cos(atd)*np.cos(PA)- cos_i*np.sin(atd)*np.sin(PA))
    vcoords_eff = pbl*(np.cos(atd)*np.sin(PA)+ cos_i*np.sin(atd)*np.cos(PA))
    pbl_eff = np.sqrt(ucoords_eff**2+vcoords_eff**2)
    return 


def _convert_orbital_radius_to_parallax(orbital_radius: Quantity,
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
    return (1*u.rad).to(u.mas)*(orbital_radius.to(u.m)/distance.to(u.m))


def _convert_parallax_to_orbital_radius(parallax: Quantity,
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
    return (parallax*distance.to(u.m))/(1*u.rad).to(u.mas)


def _calculate_stellar_radius(luminosity_star: Quantity,
                              effective_temperature: Quantity) -> Quantity:
    """Calculates the stellar radius [astropy.units.m] from its attributes.
    Only for 'delta_component' functionality

    Parameters
    ----------
    luminosity_star: astropy.units.Quantity
        The luminosity of the star [astropy.units.W]
    effective_temperature: astropy.units.Quantity
        The effective temperature of the star [astropy.units.K]

    Returns
    -------
    stellar_radius: astropy.units.quantity
        the star's radius [astropy.units.m]
    """
    return np.sqrt(luminosity_star/(4*np.pi*c.sigma_sb*effective_temperature**4))


# TODO: Make test with Jozsef's values
def stellar_flux(wavelength: Quantity,
                 effective_temperature: Quantity,
                 distance: Quantity,
                 luminosity_star: Quantity) -> Quantity:
    """Calculates the stellar flux from the distance and its radius.
    Only for 'delta_component' functionality

    Parameters
    ----------
    wavelength: astropy.units.Quantity
        The wavelength to be used for the BlackBody calculation [astropy.units.um]
    luminosity_star: astropy.units.Quantity
        The luminosity of the star [astropy.units.W]
    effective_temperature: astropy.units.Quantity
        The effective temperature of the star [astropy.units.K]
    distance: astropy.units.Quantity
        The distance to the star from the observer [astropy.units.pc]

    Returns
    -------
    stellar_flux: astropy.units.Quantity
        The star's flux [astropy.units.Jy]
    """
    plancks_law = models.BlackBody(temperature=effective_temperature)
    spectral_radiance = plancks_law(wavelength).to(u.erg/(u.cm**2*u.Hz*u.s*u.mas**2))
    stellar_radius = _calculate_stellar_radius(luminosity_star, effective_temperature)
    # TODO: Check if that can be used in this context -> The conversion
    stellar_radius_angular = _convert_orbital_radius_to_parallax(stellar_radius, distance)
    return (spectral_radiance*np.pi*(stellar_radius_angular)**2).to(u.Jy)


def _calculate_sublimation_radius(inner_temperature: Quantity,
                                  distance: Quantity,
                                  luminosity_star: Quantity) -> Quantity:
    """Calculates the sublimation radius at the inner rim of the disc

    Returns
    -------
    sublimation_radius: astropy.units.Quantity
        The sublimation radius [astropy.units.mas]
    distance: astropy.units.Quantity
        The distance to the star from the observer [astropy.units.pc]
    luminosity_star: astropy.units.Quantity
        The luminosity of the star [astropy.units.W]
    """
    radius = np.sqrt(luminosity_star/(4*np.pi*c.sigma_sb*inner_temperature**4))
    return _convert_orbital_radius_to_parallax(radius, distance)


def _calculate_sublimation_temperature(inner_radius: Quantity,
                                       distance: Quantity,
                                       luminosity_star: Quantity) -> Quantity:
    """Calculates the sublimation temperature at the inner rim of the disc

    Parameters
    ----------
    inner_radius: astropy.units.Quantity
        The inner radius of the disc [astropy.units.mas]
    luminosity_star: astropy.units.Quantity
        The luminosity of the star [astropy.units.W]

    Returns
    -------
    sublimation_temperature: astropy.units.Quantity
        The sublimation temperature [astropy.units.K]
    """
    if inner_radius.unit == u.mas:
        inner_radius = _convert_parallax_to_orbital_radius(inner_radius, distance)
    return (luminosity_star/(4*np.pi*c.sigma_sb*inner_radius**2))**(1/4)


def temperature_gradient(radius: Quantity, power_law_exponent: float,
                         inner_radius: Quantity,
                         inner_temperature: Quantity) -> Quantity:
    """Calculates the temperature gradient

    Parameters
    ----------
    radius: astropy.units.Quantity
        An array containing all the points for the radius extending outwards
        [astropy.units.mas]
    power_law_exponent: float
        A float specifying the power law exponent of the temperature gradient "q"
    inner_radius: astropy.units.Quantity
        The inner radius of the object, if not given then the sublimation radius is
        used [astropy.units.mas]
    inner_temperature: astropy.units.Quantity
        The temperature of the inner rim [astropy.units.K]

    Returns
    -------
    temperature_gradient: astropy.units.Quantity
        The temperature gradient [astropy.units.K]
    """
    temperature = models.PowerLaw1D().evaluate(radius, inner_temperature,
                                               inner_radius, power_law_exponent)
    temperature[temperature == np.inf] = 0.*temperature.unit
    return temperature


def optical_depth_gradient(radius: Quantity,
                           power_law_exponent: float, inner_radius: Quantity,
                           inner_optical_depth: Quantity) -> Quantity:
    """Calculates the optical depth gradient

    Parameters
    ----------
    radius: astropy.units.Quantity
        An array containing all the points for the radius extending outwards
        [astropy.units.mas]
    power_law_exponent: float
        A float specifying the power law exponent of the temperature gradient "q"
    inner_radius: astropy.units.Quantity
        The inner radius of the object, if not given then the sublimation radius is
        used [astropy.units.mas]
    inner_optical_depth: Quantity
        The optical depth at the inner radius [astropy.units.dimensionless_unscaled]

    Returns
    -------
    """
    optical_depth = models.PowerLaw1D().evaluate(radius, inner_optical_depth,
                                                 inner_radius, power_law_exponent)
    optical_depth[optical_depth == np.inf] = 0.*optical_depth.unit
    return optical_depth


def flux_per_pixel(wavelength: Quantity, temperature_distribution: Quantity,
                   optical_depth: Quantity, pixel_scaling: Quantity) -> Quantity:
    """Calculates the total flux of the model

    Parameters
    ----------
    wavelength: astropy.units.Quantity
        The wavelength to be used for the BlackBody calculation [astropy.units.um]
    temperature: astropy.units.Quantity
        The temperature distribution of the disc [astropy.units.K]
    optical_depth: astropy.units.Quantity
        The optical depth of the disc [astropy.units.dimensionless_unscaled]
    pixel_scaling: astropy.units.Quantity
        The pixel scaling of the field of view [astropy.units.mas]/px

    Returns
    -------
    flux: astropy.units.Quantity
        The object's flux per pixel [astropy.units.Jy]/px
    """
    plancks_law = models.BlackBody(temperature=temperature_distribution)
    # NOTE: Converts sr to mas**2. Field of view = sr or mas**2
    spectral_radiance = plancks_law(wavelength).to(u.erg/(u.cm**2*u.Hz*u.s*u.mas**2))
    flux_per_pixel = spectral_radiance*pixel_scaling**2
    return (flux_per_pixel.to(u.Jy))*(1-np.exp(-optical_depth))

################################ GENERAL UTILITY #########################################

class IterNamespace(SimpleNamespace):
    """Contains the functionality of a SimpleNamespace with the addition of a '_fields'
    attribute and the ability to iterate over the values of the '__dict__'"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fields = tuple(attr for attr in self.__dict__.keys())

    def __len__(self):
        return len(self._fields)

    def __getitem__(self, __index):
        keys = self._fields[__index]
        values = [value for value in self.__iter__()][__index]
        if isinstance(values, list):
            return IterNamespace(**dict(zip(keys, values)))
        else:
            return values

    def __iter__(self):
        for attr, val in self.__dict__.items():
            if not (attr.startswith("__") or attr.startswith("_")):
                yield val

    def to_string(self):
        values = []
        for value in self.__iter__():
            if isinstance(value, u.Quantity):
                value = np.array2string(value)
            elif isinstance(value, np.ndarray):
                value = np.array2string(value)
            elif isinstance(value, list):
                value = np.array2string(np.array(value))
            else:
                value = str(value)
            values.append(value)
        return values

    def to_string_dict(self):
        return dict(zip(self._fields, self.to_string()))


def _make_axis(axis_end: int, steps: int):
    """Makes an axis from a negative to a postive value, with the endpoint removed to give
    an even signal for a Fourier transform

    Parameters
    ----------
    axis_end: int
        The negative/positive endpoint
    steps: int
        The steps used to get from one end to another, should be an even multiple of
        the distance between positive and negative axis_end for fourier transforms

    Returns
    -------
    axis: np.ndarray
    """
    return np.linspace(-axis_end, axis_end, steps, endpoint=False)


def _set_zeros(image: Quantity,
               rvalue: Optional[bool] = False) -> Union[np.ndarray, Quantity]:
    """Sets an image grid to all zeros

    Parameters
    ----------
    image: astropy.units.Quantity
        The input image
    rvalue: bool, optional
        If 'True' returns the value of the Quantity as a np.ndarray

    Returns
    -------
    image_all_ones: np.ndarray | astropy.units.Quantity
        The output image with every value set to 0.
    """
    if not isinstance(image, u.Quantity):
        raise IOError("Input image must be [astropy.units.Quantity]")
    return image.value*0 if rvalue else image*0


def _set_ones(image: Quantity,
              rvalue: Optional[bool] = False) -> Union[np.ndarray, Quantity]:
    """Sets and image grid to all ones

    Parameters
    ----------
    image: np.ndarray | astropy.units.Quantity
        The input image
    rvalue: bool, optional
        If 'True' returns the value of the Quantity as a np.ndarray

    Returns
    -------
    image_all_ones: np.ndarray | astropy.units.Quantity
        The output image with every value set to 1.
    """
    if not isinstance(image, u.Quantity):
        raise IOError("Input image must be [astropy.units.Quantity]")
    image[image != 0.] = 1.*image.unit
    return image.value if rvalue else image


def rebin_image(image: Quantity, new_shape: Tuple, rfactor: Optional[bool] = False):
    """Rebins a 2D-image to a new input shape

    Parameters
    ----------
    image: astropy.units.quantity
        The image to be rebinned
    new_shape: Tuple
        The shape the image is to be rebinned to
    rfactor: bool, optional
        Returns the rebinning factor of the image

    Returns
    -------
    rebinned_image: np.ndarray | astropy.units.Quantity
        The rebinned image
    rebinning_factor: Tuple, optional
        The rebinning factor of the image
    """
    shape = (new_shape[0], image.shape[0] // new_shape[0],
             new_shape[1], image.shape[1] // new_shape[1])
    rebinned_image = image.reshape(shape).mean(-1).mean(1)
    if rfactor:
        factor = tuple(x//y for x, y in zip(image.shape, new_shape))
        return rebinned_image, factor
    return rebinned_image


def make_inital_guess_from_priors(priors: List[float]) -> List[float]:
    """Initialises a random float/list via a uniform distribution from the
    bounds provided

    Parameters
    -----------
    priors: IterNamespace
        Bounds list must be nested list(s) containing the bounds of the form
        form [lower_bound, upper_bound]

    Returns
    -------
    List[float]
        A list of the parameters corresponding to the priors. Also does not take the full
        priors but 1/4 from the edges, to avoid emcee problems
    """
    params = []
    for prior in priors:
        quarter_prior_distance = np.diff(prior)/4
        lower_bound, upper_bounds = prior[0]+quarter_prior_distance,\
            prior[1]-quarter_prior_distance
        param = np.random.uniform(lower_bound, upper_bounds)[0]
        params.append(param)
    return np.array(params)


def _make_params(params: List[float], units: List[Quantity],
                 labels: List[str]) -> SimpleNamespace:
    """Creates an IterNamespace for the params

    Parameters
    ----------
    params: List[float]
        The parameters's values
    units: List[astropy.units.Quantity]
        The parameter's units
    labels: List[str]
        The parameter's names

    Returns
    -------
    IterNamespace
        A namespace containing the parameters as astropy.units.Quantities
    """
    params = [param*units[i] for i, param in enumerate(params)]
    return IterNamespace(**dict(zip(labels, params)))


# TODO: Add docs
def _make_priors(priors: List[List[float]], units: List[Quantity],
                labels: List[str]) -> IterNamespace:
    """Makes the priors

    Parameters
    ----------
    params: List[float]
        The parameters's values
    units: List[astropy.units.Quantity]
        The parameter's units
    labels: List[str]
        The parameter's names

    Returns
    -------
    IterNamespace
        A namespace containing the priors as astropy.units.Quantities
    """
    priors = [u.Quantity(prior, unit=units[i]) for i, prior in enumerate(priors)]
    return IterNamespace(**dict(zip(labels, priors)))


# TODO: Implement tests for new functionality
def _make_component(name: str, component_name: str,
                    priors: Optional[List[List[Quantity]]] = None,
                    labels: Optional[List[str]] = None,
                    units: Optional[List[Quantity]] = None,
                    mod_priors: Optional[List[Quantity]] = None,
                    mod_params: Optional[List[Quantity]] = None,
                    params: Optional[List[Quantity]] = None) -> IterNamespace:
    """Creates an IterNamespace for a component of a model

    Parameters
    ----------
    name: str
        A general descriptor to differentiate components
    component_name: str
        The model component's name that is being used
    priors: List[List[Quantity], optional
        The priors for the parameters space as astropy.units.Quantity
    labels: List[str], optional
        The parameter's names
    units: List[astropy.units.Quantity], optional
        The units corresponding to the priors as astropy.units.Quantity
    mod_priors: List[List[Quantity], optional
        The modulation priors for the parameters space as astropy.units.Quantity
    mod_params: List[astropy.units.Quantity], optional
        The modulation params as astropy.units.Quantity
    params: List[astropy.units.Quantity], optional
        The parameters's values as astropy.units.Quantity

    Returns
    -------
    IterNamespace
        An IterNamespace containing the components info and its parameters
    """
    if priors is not None:
        priors = _make_priors(priors, units, labels)
    if params is not None:
        params = _make_params(params, units, labels)

    mod_labels = ["mod_amp", "mod_angle"]
    mod_units = [u.dimensionless_unscaled, u.deg]

    if mod_priors is not None:
        mod_priors = _make_priors(mod_priors, mod_units, mod_labels)
    if mod_params is not None:
        mod_params = _make_params(mod_params, mod_units, mod_labels)

    keys = ["name", "component", "params",
            "priors", "mod_params", "mod_priors"]
    values = [name, component_name, params,
              priors, mod_params, mod_priors]
    return IterNamespace(**dict(zip(keys, values)))


# TODO: Add docs
def make_fixed_params(field_of_view: int, image_size: int,
                      sublimation_temperature: int,
                      effective_temperature: int,
                      distance: int, luminosity_star: int,
                      pixel_sampling: Optional[int] = None) -> IterNamespace:
    """Crates a dictionary of the fixed params

    Parameters
    ----------
    field_of_view: int
    image_size: int
    sublimation_temperature: int
    effective_temperature: int
    distance: int
    luminosity_star: int
    pixel_sampling: int, optional
    """
    keys = ["fov", "image_size", "sub_temp", "eff_temp",
              "distance", "lum_star", "pixel_sampling"]
    values = [field_of_view, image_size, sublimation_temperature,
              effective_temperature, distance, luminosity_star, pixel_sampling]
    units = [u.mas, u.dimensionless_unscaled, u.K, u.K,
             u.pc, u.W, u.dimensionless_unscaled]
    fixed_param_dict = dict(zip(keys, values))

    for i, (key, value) in enumerate(fixed_param_dict.items()):
        if not isinstance(value, u.Quantity):
            if key == "image_size":
                fixed_param_dict[key] = u.Quantity(value, unit=u.dimensionless_unscaled,
                                                   dtype=int)
            elif key == "pixel_sampling":
                if value is None:
                    fixed_param_dict[key] = fixed_param_dict["image_size"]
                else:
                    fixed_param_dict[key] = u.Quantity(value,
                                                        unit=u.dimensionless_unscaled,
                                                        dtype=int)
            elif key == "lum_star":
                fixed_param_dict[key] *= c.L_sun
            else:
                fixed_param_dict[key] *= units[i]
        elif value.unit != units[i]:
            raise IOError(f"Wrong unit has been input for {keys[i]}. Needs to"\
                          f" be in {units[i]} or unitless!")
        else:
            continue
    return IterNamespace(**fixed_param_dict)


# TODO: Add docs and tests
def make_ring_component(name: str, priors: List[List[float]] = None,
                        mod_priors: List[float] = None,
                        mod_params: List[Quantity] = None,
                        params: List[Quantity] = None) -> IterNamespace:
    """The specific makeup of the ring-component"""
    component_name = "ring"
    labels = ["axis_ratio", "pa", "inner_radius", "outer_radius"]
    units = [u.dimensionless_unscaled, u.deg, u.mas, u.mas]
    return _make_component(name, component_name, priors, labels, units,
                          mod_priors, mod_params, params)


def make_delta_component(name: str):
    return _make_component(name, component_name="delta")

def make_gaussian_component(name: str, priors: List[List[float]] = None,
                            params: List[Quantity] = None) -> IterNamespace:
    component_name = "gaussian"
    labels, units = ["fwhm"], [u.mas]
    return _make_component(name, component_name, priors, labels, units, params=params)


# TODO: Make test for this function. Seems broken?
if __name__ == "__main__":
    lst = [(5, 10), (4, 8), (7, 6)]*u.m
    print(calculate_effective_baselines(lst, 0.5*u.dimensionless_unscaled,
                                        (180*u.deg).to(u.rad)))
