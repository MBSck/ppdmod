import numpy as np
import astropy.units as u
import astropy.constants as c

from astropy.modeling import models
from types import SimpleNamespace
from typing import List, Union, Optional
from astropy.units import Quantity


class IterNamespace(SimpleNamespace):
    """Contains the functionality of a SimpleNamespace with the addition of a '_fields'
    attribute and the ability to iterate over the values of the '__dict__'"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fields = tuple(attr for attr in self.__dict__.keys())

    def __len__(self):
        return len(self._fields)

    def __getitem__(self, index):
        keys = self._fields[index]
        values = [value for value in self.__iter__()][index]
        if isinstance(values, list):
            return IterNamespace(**dict(zip(keys, values)))
        else:
            return values


    def __iter__(self):
        for attr, val in self.__dict__.items():
            if not (attr.startswith("__") or attr.startswith("_")):
                yield val


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
    if not isinstance(orbital_radius, u.Quantity):
        orbital_radius *= u.m
    elif orbital_radius.unit not in [u.m, u.au]:
        raise IOError("Wrong unit has been input, orbital radius needs to be in"\
                      "[astropy.units.m], [astropy.units.au] or unitless!")

    if not isinstance(distance, u.Quantity):
        distance *= u.pc
    elif distance.unit != u.pc:
        raise IOError("Wrong unit has been input, distance needs to be in"\
                      "[astropy.units.pc] or unitless!")

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
    if not isinstance(parallax, u.Quantity):
        parallax *= u.mas
    elif parallax.unit != u.mas:
        raise IOError("Wrong unit has been input, parallax needs to be in"\
                      "[astropy.units.mas] or unitless!")

    if not isinstance(distance, u.Quantity):
        distance *= u.pc
    elif distance.unit != u.pc:
        raise IOError("Wrong unit has been input, distance needs to be in"\
                      "[astropy.units.pc] or unitless!")

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
    if not isinstance(wavelength, u.Quantity):
        wavelength = (wavelength*u.um).to(u.AA)
    elif wavelength.unit == u.um:
        wavelength = wavelength.to(u.AA)
    else:
        raise IOError("Enter the wavelength in [astropy.units.um] or unitless!")

    if not isinstance(effective_temperature, u.Quantity):
        effective_temperature *= u.K
    elif effective_temperature.unit != u.K:
        raise IOError("Enter the effective temperature in [astropy.units.K] or unitless!")

    if not isinstance(distance, u.Quantity):
        distance *= u.pc
    elif distance.unit != u.pc:
        raise IOError("Enter the distance in [astropy.units.pc] or unitless!")

    if not isinstance(luminosity_star, u.Quantity):
        luminosity_star *= u.W
    elif luminosity_star.unit != u.W:
        raise IOError("Enter the luminosity of the star in [astropy.units.W]"\
                      " or unitless!")

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
    if not isinstance(inner_temperature, u.Quantity):
        inner_temperature *= u.K
    elif inner_temperature.unit != u.K:
        raise IOError("Enter the inner temperature in [astropy.units.K] or unitless!")

    if not isinstance(distance, u.Quantity):
        distance *= u.pc
    elif distance.unit != u.pc:
        raise IOError("Enter the distance in [astropy.units.pc] or unitless!")

    if not isinstance(luminosity_star, u.Quantity):
        luminosity_star *= u.W
    elif luminosity_star.unit != u.W:
        raise IOError("Enter the luminosity of the star in [astropy.units.W]"\
                      " or unitless!")

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
    if not isinstance(distance, u.Quantity):
        distance *= u.pc
    elif distance.unit != u.pc:
        raise IOError("Enter the distance in [astropy.units.pc] or unitless!")

    if not isinstance(luminosity_star, u.Quantity):
        luminosity_star *= u.W
    elif luminosity_star.unit != u.W:
        raise IOError("Enter the luminosity of the star in [astropy.units.W]"\
                      " or unitless!")
    if inner_radius.unit == u.mas:
        inner_radius = _convert_parallax_to_orbital_radius(inner_radius, distance)
    elif inner_radius.unit != u.mas:
        raise IOError("Enter the inner radius in [astropy.units.mas]"\
                      " or unitless!")

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

    Returns
    -------
    temperature_gradient: astropy.units.Quantity
        The temperature gradient [astropy.units.K]
    """
    if not isinstance(radius, u.Quantity):
        radius *= u.mas
    elif radius.unit != u.mas:
        raise IOError("Enter the radius in [astropy.units.mas] or unitless!")

    if not isinstance(power_law_exponent, u.Quantity):
        power_law_exponent *= u.dimensionless_unscaled
    elif power_law_exponent.unit != u.dimensionless_unscaled:
        raise IOError("Enter the inner temperature in"\
                      " [astropy.units.dimensionless_unscaled] or unitless!")

    if not isinstance(inner_radius, u.Quantity):
        inner_radius *= u.mas
    elif inner_radius.unit != u.mas:
        raise IOError("Enter the inner radius in [astropy.units.mas] or unitless!")

    if not isinstance(inner_temperature, u.Quantity):
        inner_temperature *= u.K
    elif inner_temperature.unit != u.K:
        raise IOError("Enter the inner temperature in [astropy.units.K] or unitless!")


    return models.PowerLaw1D().evaluate(radius, inner_temperature,
                                        inner_radius, power_law_exponent)


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
    if not isinstance(radius, u.Quantity):
        radius *= u.mas
    elif radius.unit != u.mas:
        raise IOError("Enter the radius in [astropy.units.mas] or unitless!")

    if not isinstance(power_law_exponent, u.Quantity):
        power_law_exponent *= u.dimensionless_unscaled
    elif power_law_exponent.unit != u.dimensionless_unscaled:
        raise IOError("Enter the inner temperature in"\
                      " [astropy.units.dimensionless_unscaled] or unitless!")

    if not isinstance(inner_radius, u.Quantity):
        inner_radius *= u.mas
    elif inner_radius.unit != u.mas:
        raise IOError("Enter the inner radius in [astropy.units.mas] or unitless!")

    if not isinstance(inner_optical_depth, u.Quantity):
        inner_optical_depth *= u.dimensionless_unscaled
    elif inner_optical_depth.unit != u.dimensionless_unscaled:
        raise IOError("Enter the inner optical depth in"\
                      " [astropy.units.dimensionless_unscaled] or unitless!")

    return models.PowerLaw1D().evaluate(radius, inner_optical_depth,
                                        inner_radius, power_law_exponent)


# TODO: Fix the tests here
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
    if not isinstance(wavelength, u.Quantity):
        wavelength = (wavelength*u.um).to(u.AA)
    elif wavelength.unit == u.um:
        wavelength = wavelength.to(u.AA)
    else:
        raise IOError("Enter the wavelength in [astropy.units.um] or unitless!")

    if not isinstance(temperature_distribution, u.Quantity):
        temperature_distribution *= u.K
    elif temperature_distribution.unit != u.K:
        raise IOError("Enter the temperature distribution in"\
                      " [astropy.units.K] or unitless!")

    if not isinstance(optical_depth, u.Quantity):
        optical_depth *= u.dimensionless_unscaled
    elif optical_depth.unit != u.dimensionless_unscaled:
        raise IOError("Enter the optical depth distribution in"\
                      " [astropy.units.dimensionless_unscaled] or unitless!")

    if not isinstance(pixel_scaling, u.Quantity):
        pixel_scaling *= u.mas
    elif pixel_scaling.unit != u.mas:
        raise IOError("Enter the pixel scaling in [astropy.units.mas] or unitless!")

    plancks_law = models.BlackBody(temperature=temperature_distribution)
    # NOTE: Convert sr to mas**2. Field of view = sr or mas**2
    spectral_radiance = plancks_law(wavelength).to(u.erg/(u.cm**2*u.Hz*u.s*u.mas**2))
    flux_per_pixel = spectral_radiance*pixel_scaling**2
    return (flux_per_pixel.to(u.Jy))*(1-np.exp(-optical_depth))


def _set_units_for_priors(priors: List, units: List[Quantity]) -> List[Quantity]:
    """Sets the [astropy.units] for the priors

    Parameters
    ----------
    priors: List
        The priors
    units: List[Quantity]
        The to the priors corresponding units

    Returns
    -------
    priors: List[Quantity]
        A list containing the nested [astropy.units.Quantity] that are the priors
    """
    return [u.Quantity(prior, unit=units[i]) for i, prior in enumerate(priors)]

def _set_params_from_priors(priors: List[float]) -> List[float]:
    """Initialises a random float/list via a uniform distribution from the
    bounds provided

    Parameters
    -----------
    priors: IterNamespace
        Bounds list must be nested list(s) containing the bounds of the form
        form [lower_bound, upper_bound]

    Returns
    -------
    List[astropy.units.Quantity]
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
    return params


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
        An IterNamespace made up from the names and values of the params
    """
    if not all(isinstance(param, u.Quantity) for param in params):
        params = [param*units[i] for i, param in enumerate(params)]
    return IterNamespace(**dict(zip(labels, params)))


# TODO: Add docs
def _make_priors(priors: List[List[float]], units: List[Quantity],
                labels: List[str]) -> IterNamespace:
    """Makes the priors"""
    return _make_params(_set_units_for_priors(priors, units), units, labels)


def _make_params_from_priors(priors: IterNamespace,
                             units: List[Quantity], labels: List[str]) -> IterNamespace:
    """Makes params from priors"""
    return _make_params(_set_params_from_priors(priors), units, labels)


# TODO: Implement tests for new functionality
def _make_component(name: str, component_name: str,
                    priors: List[List[Quantity]] = None,
                    labels: List[str] = None,
                    units: List[Quantity] = None,
                    mod_priors: List[Quantity] = None,
                    mod_params: List[Quantity] = None,
                    params: List[Quantity] = None) -> IterNamespace:
    """Creates an IterNamespace for a component of a model

    Parameters
    ----------
    name: str
        A general descriptor to differentiate components
    component_name: str
        The model component's name that is being used
    priors: List[List[Quantity], optional
        The priors for the parameters space
    labels: List[str], optional
        The parameter's names
    units: List[astropy.units.Quantity], optional
        The units corresponding to the priors
    mod_priors: List[List[Quantity], optional
    mod_params: List[astropy.units.Quantity], optional
    params: List[astropy.units.Quantity], optional
        The parameters's values

    Returns
    -------
    IterNamespace
        An IterNamespace containing the components info and its params
    """
    if priors is not None:
        priors = _make_priors(priors, units, labels)
    if (params is None) and (priors is not None):
        params = _make_params_from_priors(priors, units, labels)
    if params is not None:
        params = _make_params(params, units, labels)

    mod_labels = ["mod_amp", "mod_angle"]
    mod_units = [u.dimensionless_unscaled, u.deg]

    if mod_priors is not None:
        mod_priors = _make_priors(mod_priors, mod_units, mod_labels)
    if (mod_params is None) and (mod_priors is not None):
        mod_params = _make_params_from_priors(mod_priors, mod_units, mod_labels)
    elif mod_params is not None:
        mod_params = _make_params(mod_params, mod_units, mod_labels)

    keys = ["name", "component", "params",
            "priors", "mod_params", "mod_priors"]
    values = [name, component_name, params,
              priors, mod_params, mod_priors]
    return IterNamespace(**dict(zip(keys, values)))


def make_disc_params(priors: List, params: Optional[List] = None) -> IterNamespace:
    """Makes the disc param's tuple"""
    labels, units = ["q", "p"], [u.dimensionless_unscaled, u.dimensionless_unscaled]
    priors = _make_priors(priors, units, labels)

    for i, prior in enumerate(priors):
        if not isinstance(prior, u.Quantity):
            priors[i] = prior*units[i]
        elif prior.unit != units[i]:
            raise IOError(f"Wrong unit has been input for priors in {keys[i]}. Needs to"\
                          f" be in {units[i]} or unitless!")
        if params:
            if not isinstance(params[i], u.Quantity):
                params[i] = params[i]*units[i]
            elif params[i].unit != units[i]:
                raise IOError(f"Wrong unit has been input for params in {keys[i]}."\
                              f" Needs to be in {units[i]} or unitless!")

    if params is None:
        params = _make_params_from_priors(priors, units, labels)
    else:
        params = _make_params(params, units, labels)
    keys, values = ["params", "priors"], [params, priors]
    return IterNamespace(**dict(zip(keys, values)))


# TODO: Add docs
def make_fixed_params(field_of_view: int, image_size: int,
                      sublimation_temperature: int,
                      effective_temperature: int,
                      distance: int, luminosity_star: int,
                      tau: float, pixel_sampling: Optional[int] = None) -> IterNamespace:
    """Crates a dictionary of the fixed params

    Parameters
    ----------
    field_of_view: int
    image_size: int
    sublimation_temperature: int
    effective_temperature: int
    distance: int
    luminosity_star: int
    tau: float
    pixel_sampling: int, optional
    """
    keys = ["fov", "image_size", "sub_temp", "eff_temp",
              "distance", "lum_star", "pixel_sampling", "tau"]
    values = [field_of_view, image_size, sublimation_temperature,
              effective_temperature, distance, luminosity_star, pixel_sampling, tau]
    units = [u.mas, u.dimensionless_unscaled, u.K, u.K,
             u.pc, u.W, u.dimensionless_unscaled, u.dimensionless_unscaled]
    fixed_param_dict = dict(zip(keys, values))

    for i, (key, value) in enumerate(fixed_param_dict.items()):
        if (key == "pixel_sampling") and (value is None):
            fixed_param_dict[key] = fixed_param_dict["image_size"]
        if not isinstance(value, u.Quantity):
            if key == "image_size":
                fixed_param_dict[key] = u.Quantity(value, unit=u.dimensionless_unscaled,
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


# TODO: Maybe rename this function?
def _check_attributes(params: IterNamespace,
                      attributes: List[str], units: List[Quantity]) -> bool:
    """Checks the attributes contained in an IterNamespace with them of the model class.
    Returns 'True' if the attributes exist and are in the right [astropy.units]

    Parameters
    ----------
    params: IterNamespace
        A IterNamespace containing the parameters and their values
    attributes: List[str]
        The names of the parameters
    units: List[Quantity]
        A list containing the required units for the respective parameters

    Returns
    -------
    bool
    """
    for i, parameter in enumerate(zip(params._fields, params)):
        param_name, param = parameter
        if param_name in attributes:
            if not param.unit == units[i]:
                raise IOError(f"Wrong unit for parameter '{param_name}' has been input!")
        else:
            raise IOError(f"Check inputs! {param_name} is missing!\n"\
                          f"Required inputs: {attributes}")
    return True


# TODO: Make test for this function. Seems broken?
def check_and_convert(params: Union[List[Quantity], IterNamespace],
                      attributes: List[str], units: List[Quantity]) -> IterNamespace:
    """Checks if 'params' is a IterNamespace and if not converts it to one. Also checks if
    the provided IterNamespace contains all needed parameters

    Parameters
    ----------
    params: Union[List, IterNamespace]
        Either a list or a IterNamespace of params
    attributes: List[str]
        The labels that are being checked for in the IterNamespace -> Parameter checks
    units: List[Quantity]
        The units that are being checked for in the params

    Returns
    -------
    params: IterNamespace
    """
    if isinstance(params, IterNamespace):
        # NOTE: Raises error if wrong input
        if _check_attributes(params, attributes, units):
            return params
    else:
        return _make_params(params, units, attributes)


if __name__ == "__main__":
    fixed_params = make_fixed_params(50, 128, 1500,
                                     7900*u.K, 140*u.pc, 19*c.L_sun, 1)
    print([value for value in fixed_params])
    print(fixed_params.fov.value)
    print(fixed_params[-1])

