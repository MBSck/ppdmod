import numpy as np
import astropy.units as u

from types import SimpleNamespace
from typing import List, Union, Optional
from astropy.units import Quantity


# TODO: Make sure the tests for this are solid and updated
# TODO: Make parameter checks for the tuples, that if wrong params are entered if raises
# Error

class IterNamespace(SimpleNamespace):
    """Contains the functionality of a SimpleNamespace with the addition of a '_fields'
    attribute and the ability to iterate over the values of the '__dict__'"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fields = [attr for attr in self.__dict__.keys()]


    def __iter__(self):
        for attr, val in self.__dict__.items():
            if not attr.startswith("__"):
                yield val


# TODO: Make test for this function. Seems broken?
def _check_and_convert(params: Union[List[Quantity], IterNamespace],
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
        if check_attributes(params, attributes, units):
            return params
    else:
        return make_params(params, attributes)


def make_params(params: List[Quantity], labels: List[str]) -> SimpleNamespace:
    """Creates an IterNamespace for the params

    Parameters
    ----------
    params: List[astropy.units.Quantity]
        The parameters's values
    labels: List[str]
        The parameter's names

    Returns
    -------
    IterNamespace
        An IterNamespace made up from the names and values of the params
    """
    if not all([isinstance(param, u.Quantity) for param in params]):
        raise IOError("All params have to be a [astropy.units.Quantity]!")

    return IterNamespace(**dict(zip(params, labels)))


# TODO: Add docs and tests
def make_priors(priors: List[List[float]], units: List[Quantity],
                labels: List[str]) -> IterNamespace:
    """Makes the priors"""
    return make_params(_set_units_for_priors(priors, units), labels)


def make_params_from_priors(priors: IterNamespace, labels: List[str]) -> IterNamespace:
    """Makes params from priors"""
    return make_params(_set_params_from_priors(priors), labels)


# TODO: Implement tests for new functionality
def make_component(name: str, component_name: str,
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
    priors: List[List[Quantity]
        The priors for the parameters space
    labels: List[str]
        The parameter's names
    units: List[astropy.units.Quantity]
        The units corresponding to the priors
    mod_priors
    mod_params
    params: List[astropy.units.Quantity], optional
        The parameters's values

    Returns
    -------
    IterNamespace
        An IterNamespace containg the components info and its params
    """
    if priors is not None:
        priors = make_priors(priors, units, labels)
    if (params is None) and (priors is not None):
        params = make_params_from_priors(priors, labels)
    if params is not None:
        params = make_params(params, labels)

    mod_labels = ["mod_angle", "mod_amp"]
    mod_units = [u.deg, u.dimensionless_unscaled]

    if mod_priors is None:
        modulated = False
    elif mod_params is None:
        modulated = True
        mod_priors = make_priors(mod_priors, mod_units, mod_labels)
        mod_params = make_params_from_priors(mod_priors, mod_labels)
    else:
        mod_params = make_params(mod_params, mod_labels)

    keys = ["name", "component", "modulated", "params",
            "priors", "mod_params", "mod_priors"]
    values = [name, component_name, modulated, params,
              priors, mod_params, mod_priors]
    return IterNamespace(**dict(zip(keys, values)))


# TODO: Write test for this
def make_disc_params(priors: List, prior_units: List[Quantity],
                     params: Optional[List] = None) -> IterNamespace:
    """Makes the disc param's tuple"""
    keys = ["params", "priors"]
    priors = make_params(_set_units_for_priors(priors, prior_units), labels)
    if params is None:
        params = make_params(_set_params_from_priors(priors), labels)
    else:
        params = make_params(params, labels)
    return IterNamespace(**dict(zip(keys, params)))


# TODO: Add docs and tests
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
    return IterNamespace(**dict(zip(keys, values)))


# TODO: Maybe rename this function?
def check_attributes(params: IterNamespace,
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
                raise IOError(f"Wrong unit for parameter '{param}' has been input!")
        else:
            raise IOError(f"Check inputs! {param_name} is missing!\n"\
                          f"Required inputs: {attributes}")
    return True


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


def _set_params_from_priors(priors: IterNamespace) -> Quantity:
    """Initialises a random float/list via a normal distribution from the
    bounds provided
    Parameters
    -----------
    priors: IterNamespace
        Bounds list must be nested list(s) containing the bounds of the form
        form [lower_bound, upper_bound]
    centre_rnd: bool, optional
        Get a random number close to the centre of the bound
    Returns
    -------
    astropy.units.Quantity
    """
    params = []
    # NOTE: Tries to get values in the centre
    # TODO: Maybe implement complete random values
    for prior in priors:
        quarter_prior_distance = np.diff(prior.value)/4
        lower_bound, upper_bounds = prior.value[0]+quarter_prior_distance,\
            prior.value[1]-quarter_prior_distance
        param = np.random.uniform(lower_bound, upper_bounds)
        params.append(param*prior.unit)
    return params

# TODO: Add docs and tests
def make_ring_component(name: str, priors: List[List[Quantity]] = None,
                        mod_priors: List[Quantity] = None,
                        mod_params: List[Quantity] = None,
                        params: List[Quantity] = None,
                        geometric_params: List[float] = None) -> IterNamespace:
    """The specific makeup of the ring-component"""
    component_name = "ring"
    labels = ["axis_ratio", "pa", "inner_radius", "outer_radius"]
    units = [u.dimensionless_unscaled, u.deg, u.mas, u.mas]
    return make_component(name, component_name, priors, labels, units,
                          mod_priors, mod_params, params, geometric_params)

def make_delta_component(name: str):
    return make_component(name, component_name="delta")

if __name__ == "__main__":
    fixed_params = make_fixed_params(50*u.mas, 128, 1500*u.K, 7900*u.K, 140*u.pc, None)
    print(fixed_params._fields)

