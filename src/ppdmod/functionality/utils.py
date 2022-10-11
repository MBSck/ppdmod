from os import name
import astropy.units as u

from typing import List
from astropy.units import Quantity
from collections import namedtuple


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
                         params: List[Quantity], labels: List[str]) -> namedtuple:

    """Creates a named tuple for a component of a model

    Parameters
    ----------
    component_name: str
        The component's name
    params: List[astropy.units.Quantity]
        The parameters's values
    labels: List[str]
        The parameter's names

    Returns
    -------
    namedtuple
        A named tuple containg the components info and its params
    """
    Component = namedtuple("Component", ["name", "params"])
    params = make_params_tuple(params, labels)
    return Component(component_name, params)

def check_attributes(params: namedtuple,
                     attributes: List[str], units: List[Quantity]) -> bool:
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


if __name__ == "__main__":
    ...
