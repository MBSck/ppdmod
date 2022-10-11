from os import name
import astropy.units as u

from typing import List
from astropy.units import Quantity
from collections import namedtuple


# TODO: Make this class independent?
def _check_attrs(self, named_tuple: namedtuple,
                 attributes: List[str], units: List[Quantity]) -> bool:
    """Checks the attributes contained in a named tuple of the model class. Returns
    true if the attributes exist and are in the right [astropy.units]

    Parameters
    ----------
    named_tuple: namedtuple
        A named tuple containing the parameters and their values
    attributes: List[str]
        The names of the parameters
    units: List[Quantity]
        A list containing the required units for the respective parameters

    Returns
    -------
    bool
    """
    # for i, param_name, param in enumerate(zip(named_tuple._fields, named_tuple)):
        # if
        # else:
            # raise IOError(f"Check inputs! {param_name} is missing!\n"\
                          # f"Required inputs: {attributes}")
    ...

def make_params_named_tuple(params: List[Quantity], labels: List[str]) -> namedtuple:
    """Creates a named tuple for the params

    Parameters
    ----------
    """
    if not all([isinstance(param, u.Quantity) for param in params]):
        raise IOError("All params have to be a [astropy.units.Quantity]!")

    Params = namedtuple("Params", labels)
    return Params(*params)

if __name__ == "__main__":
    labels = ["x", "y"]
    params = [10*u.mas, 5*u.dimensionless_unscaled]
    params_tuple = make_params_named_tuple(params, labels)
    print(params_tuple.x)
