import numpy as np
import astropy.units as u

from typing import List, Union
from astropy.units import Quantity
from collections import namedtuple

# TODO: Make sure the tests for this are solid and updated
# TODO: Make parameter checks for the tuples, that if wrong params are entered if raises
# Error

# TODO: Make test for this function
def _check_and_convert(params: Union[List[Quantity], namedtuple],
                       attributes: List[str],
                       units: List[Quantity]) -> namedtuple:
    """Checks if 'params' is a namedtuple and if not converts it to one. Also checks if
    the provided named tuple contains all needed parameters

    Parameters
    ----------
    params: Union[List, namedtuple]
        Either a list or a namedtuple of params
    attributes: List[str]
        The labels that are being checked for in the namedtuple -> Parameter checks
    units: List[Quantity]
        The units that are being checked for in the params

    Returns
    -------
    params: namedtuple
    """
    if isnamedtupleinstance(params):
        # NOTE: Raises error if wrong input
        if check_attributes(params, attributes, units):
            return params
    else:
        return make_params_tuple(params, attributes)

def isnamedtupleinstance(x) -> bool:
    """Code from
    https://stackoverflow.com/questions/2166818/how-to-check-if-an-object-is-an-instance-of-a-namedtuple"""
    t = type(x)
    b = t.__bases__
    if len(b) != 1 or b[0] != tuple: return False
    f = getattr(t, '_fields', None)
    if not isinstance(f, tuple): return False
    return all(type(n)==str for n in f)


def make_params_tuple(params: List[Quantity], labels: List[str]) -> namedtuple:
    """Creates a named tuple for the params

    Parameters
    ----------
    params: List[astropy.units.Quantity]
        The parameters's values
    labels: List[str]
        The parameter's names

    Returns
    -------
    namedtuple
        A named tuple made up from the names and values of the params
    """
    if not all([isinstance(param, u.Quantity) for param in params]):
        raise IOError("All params have to be a [astropy.units.Quantity]!")

    Params = namedtuple("Params", labels)
    return Params(*params)


def make_component_tuple(component_name: str,
                         priors: List[List[Quantity]],
                         labels: List[str],
                         units: List[Quantity],
                         params: List[Quantity] = None) -> namedtuple:
    """Creates a named tuple for a component of a model

    Parameters
    ----------
    component_name: str
        The component's name
    priors: List[List[Quantity]
        The priors for the parameters space
    labels: List[str]
        The parameter's names
    units: List[astropy.units.Quantity]
        The units corresponding to the priors
    params: List[astropy.units.Quantity], optional
        The parameters's values

    Returns
    -------
    namedtuple
        A named tuple containg the components info and its params
    """
    Component = namedtuple("Component", ["name", "params", "priors"])
    priors = make_params_tuple(_set_units_for_priors(priors, units), labels)
    if params is None:
        params = make_params_tuple(_set_params_from_priors(priors), labels)
    else:
        params = make_params_tuple(params, labels)
    return Component(component_name, params, priors)


# TODO: Maybe rename this function?
def check_attributes(params: namedtuple,
                     attributes: List[str], units: List[Quantity]) -> None:
    """Checks the attributes contained in a named tuple of the model class. Returns
    true if the attributes exist and are in the right [astropy.units]

    Parameters
    ----------
    params: namedtuple
        A named tuple containing the parameters and their values
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


def _set_params_from_priors(priors: namedtuple) -> Quantity:
    """Initialises a random float/list via a normal distribution from the
    bounds provided
    Parameters
    -----------
    priors: namedtuple
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
        centre_of_prior = np.diff(prior.value)
        half_distance_from_centre = centre_of_prior/4
        param = np.random.normal(centre_of_prior, half_distance_from_centre)
        params.append(param*prior.unit)
    return params


if __name__ == "__main__":
    ...
