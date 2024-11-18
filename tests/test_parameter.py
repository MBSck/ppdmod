import astropy.units as u
import numpy as np
import pytest
from numpy.typing import ArrayLike

from ppdmod.parameter import Parameter

VALUE = np.arange(0, 10) * u.mas
WAVELENGTH = np.linspace(8, 13, 10, endpoint=False) * u.um
WAVELENGTHS_AND_VALUES = list(zip(WAVELENGTH, VALUE))


@pytest.fixture
def x() -> Parameter:
    """Parameter x."""
    return Parameter(base="x")


@pytest.fixture
def x_filled() -> Parameter:
    return Parameter(value=10, min=0, max=100, free=True, base="x")


def test_parameter(x: Parameter) -> None:
    """Check the parameters attributes"""
    assert x.name == "x"
    assert x.value == 0
    assert x.unit == u.mas
    assert x() == 0 * u.mas
    assert not x.free


@pytest.mark.parametrize(
    "array, expected",
    [
        (1, 1),
        (
            (
                5,
                10,
            ),
            np.array([5, 10]),
        ),
        ([6, 7], np.array([6, 7])),
        ([8, 12] * u.um, [8, 12] * u.um),
    ],
)
def test_set_to_numpy_array(
    x: Parameter, array: ArrayLike, expected: ArrayLike
) -> None:
    """Tests if an array gets converted to a numpy array."""
    converted_array = x._set_to_numpy_array(array)
    assert np.array_equal(converted_array, expected)


@pytest.mark.parametrize("wavelength, expected", WAVELENGTHS_AND_VALUES)
def test_individual_wavelength_calling(
    x: Parameter, wavelength: float, expected: int
) -> None:
    """Tests the __call__ of the Parameter class for
    singular wavelengths."""
    x.value, x.grid = VALUE, WAVELENGTH
    assert x(wavelength) == expected


@pytest.mark.parametrize(
    "wavelength, expected",
    zip(
        np.linspace(8.25, 13.25, 5, endpoint=False) * u.um,
        (np.linspace(5, 85, 5) / 10) * u.mas,
    ),
)
def test_interpolation(x: Parameter, wavelength: u.um, expected: u.mas) -> None:
    """Tests the interpolation function."""
    x.interpolate = True
    x.value, x.grid = VALUE, WAVELENGTH
    assert x(wavelength) == expected


def test_multiple_wavelength_calling(x: Parameter) -> None:
    """Tests the __call__ of the Parameter class for
    multiple wavelengths."""
    x.value, x.grid = VALUE, WAVELENGTH
    assert np.allclose(x(WAVELENGTH), VALUE)


def test_process_base(x_filled: Parameter) -> None:
    """Tests the setting of a base class without overriding given values."""
    x_filled = Parameter(value=10, min=0, max=100, free=True, base="x")
    assert x_filled.free
    assert x_filled.value == 10
    assert x_filled() == 10 * u.mas
    assert x_filled.min == 0
    assert x_filled.max == 100


def test_get_limits(x: Parameter):
    assert x.get_limits() == (-30, 30)


def test_post_init(x_filled: Parameter): ...


def test_copy(x: Parameter):
    x_new = x.copy()
    assert x == x_new

    x_new.value = 5.33
    assert x() != x_new()
