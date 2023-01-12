from types import SimpleNamespace
from typing import Any, List, Tuple, Union, Optional
from dataclasses import dataclass

import numpy as np
import astropy.units as u
import astropy.constants as const


    fixed_params = make_fixed_params(50, 128, 1500, 7900, 140, 19)
@dataclass
class FixedParams:
    """Class that contains the fixed parameters for the modelling."""
    field_of_view: int
    image_size_px: int
    sublimation_temperature: int
    effective_temperature: int
    distance_to_star: float
    luminosity_star: float

class IterNamespace(SimpleNamespace):
    """Contains the functionality of a SimpleNamespace with the addition of a '_fields'
    attribute and the ability to iterate over the values of the '__dict__'"""
    def __init__(self, *args: Any, **kwargs: Any):
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


def _set_zeros(image: u.Quantity,
               rvalue: Optional[bool] = False) -> Union[np.ndarray, u.Quantity]:
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


def _set_ones(image: u.Quantity,
              rvalue: Optional[bool] = False) -> Union[np.ndarray, u.Quantity]:
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


def rebin_image(image: u.Quantity, new_shape: Tuple,
                rfactor: Optional[bool] = False) -> u.Quantity:
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


def _make_params(params: List[float], units: List[u.Quantity],
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
def _make_priors(priors: List[List[float]], units: List[u.Quantity],
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
                    priors: Optional[List[List[u.Quantity]]] = None,
                    labels: Optional[List[str]] = None,
                    units: Optional[List[u.Quantity]] = None,
                    mod_priors: Optional[List[u.Quantity]] = None,
                    mod_params: Optional[List[u.Quantity]] = None,
                    params: Optional[List[u.Quantity]] = None) -> IterNamespace:
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
                fixed_param_dict[key] *= const.L_sun
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
                        mod_params: List[u.Quantity] = None,
                        params: List[u.Quantity] = None) -> IterNamespace:
    """The specific makeup of the ring-component"""
    component_name = "ring"
    labels = ["axis_ratio", "pa", "inner_radius", "outer_radius"]
    units = [u.dimensionless_unscaled, u.deg, u.mas, u.mas]
    return _make_component(name, component_name, priors, labels, units,
                          mod_priors, mod_params, params)


def make_delta_component(name: str):
    return _make_component(name, component_name="delta")

def make_gaussian_component(name: str, priors: List[List[float]] = None,
                            params: List[u.Quantity] = None) -> IterNamespace:
    component_name = "gaussian"
    labels, units = ["fwhm"], [u.mas]
    return _make_component(name, component_name, priors, labels, units, params=params)


# TODO: Make test for this function. Seems broken?
if __name__ == "__main__":
    lst = [(5, 10), (4, 8), (7, 6)]*u.m
    res = calculate_effective_baselines(lst, 0.5*u.dimensionless_unscaled,
                                        (180*u.deg).to(u.rad), 8*u.um)
    breakpoint()
