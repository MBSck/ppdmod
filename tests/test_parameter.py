import astropy.units as u
import numpy as np
import pytest
from numpy.typing import ArrayLike

from ppdmod.options import STANDARD_PARAMETERS
from ppdmod.parameter import Parameter


VALUE = np.arange(0, 10)*u.mas
WAVELENGTH = np.linspace(8, 13, 10, endpoint=False)*u.um
WAVELENGTHS_AND_VALUES = list(zip(WAVELENGTH, VALUE))


@pytest.fixture
def x() -> Parameter:
    """Parameter x."""
    return Parameter(**STANDARD_PARAMETERS["x"])


def test_parameter(x: Parameter) -> None:
    """Check the parameters attributes"""
    assert x.name == "x"
    assert x.value == 0
    assert x.unit == u.mas
    assert x() == 0*u.mas
    assert not x.free


def test_set_parameter_limits(x: Parameter) -> None:
    """Check set limit function"""
    x.set(min=5, max=10)
    assert x.min == 5 and x.max == 10


@pytest.mark.parametrize(
        "array, expected", [(1, 1), ((5, 10,), np.array([5, 10])),
                            ([6, 7], np.array([6, 7])),
                            ([8, 12]*u.um, [8, 12]*u.um)])
def test_set_to_numpy_array(
        x: Parameter, array: ArrayLike, expected: ArrayLike) -> None:
    """Tests if an array gets converted to a numpy array."""
    converted_array = x._set_to_numpy_array(array)
    assert np.array_equal(converted_array, expected)


@pytest.mark.parametrize("wavelength, expected", WAVELENGTHS_AND_VALUES)
def test_individual_wavelength_calling(
        x: Parameter, wavelength: float, expected: int) -> None:
    """Tests the __call__ of the Parameter class for
    singular wavelengths."""
    x.value, x.wavelength = VALUE, WAVELENGTH
    assert x(wavelength) == expected


@pytest.mark.parametrize(
        "wavelength, expected", zip(np.linspace(8.25, 13.25, 5, endpoint=False)*u.um,
                                    (np.linspace(5, 85, 5)/10)*u.mas))
def test_interpolation(
        x: Parameter, wavelength: u.um, expected: u.mas) -> None:
    """Tests the interpolation function."""
    x.interpolate = True
    x.value, x.wavelength = VALUE, WAVELENGTH
    assert x(wavelength) == expected


def test_multiple_wavelength_calling(x: Parameter) -> None:
    """Tests the __call__ of the Parameter class for
    multiple wavelengths."""
    x.value, x.wavelength = VALUE, WAVELENGTH
    assert np.allclose(x(WAVELENGTH), VALUE)
