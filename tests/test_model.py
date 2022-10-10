import pytest
import numpy as np

import astropy.units as u
import astropy.constants as c

from ppdmod.functionality.model import Model

# TODO: Improve a lot of theses tests and check how to test properly

################################### Fixtures #############################################

@pytest.fixture
def mock_init_values():
    """The sublimation temp [astropy.units.K], the eff_temp [astropy.units.K], the
    distance of the star [astropy.units.pc], the luminosity [astropy.units.L_sun] and
    the wavelength [u.um]"""
    return [50, 128, 1500, 7900, 140, 19]

@pytest.fixture
def mock_init_values_all_one():
    return [1, 1, 1, 1, 1, 1]

@pytest.fixture
def mock_values_hd142666():
    low_bound = [50, 128, 1500, 7900, 120, 13]
    middle = [50, 128, 1500, 7900, 140, 19]
    high_bound = [50, 128, 1500, 7900, 160, 25]
    return low_bound, middle, high_bound

@pytest.fixture
def wavelength():
    return 8*u.um

@pytest.fixture
def incline_params():
    return [0.6, 180]

################################ ReadoutFits - TESTS #####################################

# TODO: Write tests for setters and getters

def test_init(mock_init_values):
    # TODO: Maybe extend this text
    assert Model(*mock_init_values)

def test_convert_orbital_radius_from_parallax(mock_init_values_all_one):
    # TODO: Make test work for the other way round
    model = Model(*mock_init_values_all_one)
    orbital_radius = model._convert_orbital_radius_from_parallax(1e3*u.mas)
    model.distance = 100
    orbital_radius_hundred_pc = model._convert_orbital_radius_from_parallax(1e3*u.mas)
    assert orbital_radius.unit == u.m
    assert orbital_radius.value == c.au.value
    # FIXME: Is this test correct?
    assert (orbital_radius_hundred_pc.value <= (1e2*c.au.value + 0.1))\
        and (orbital_radius_hundred_pc.value > (1e2*c.au.value - 1e6))

def test_convert_parallax_from_orbital_radius(mock_init_values_all_one):
    # TODO: Make test work for the other way round
    model = Model(*mock_init_values_all_one)
    orbital_radius = model._convert_parallax_from_orbital_radius(1*u.au)
    model.distance = 100
    orbital_radius_hundred_pc = model._convert_parallax_from_orbital_radius(1*u.au)
    assert orbital_radius.unit == u.mas
    # These values have been adapted for u.mas instead of u.arcsec
    assert (orbital_radius.value <= 1e3)\
        and (orbital_radius.value > (1e3 - 0.1))
    assert (orbital_radius_hundred_pc.value < 0.01e3)\
        and (orbital_radius_hundred_pc.value > (0.01e3 - 0.1))

def test_stellar_radius():
    ...

def test_calculate_sublimation_temperature(mock_init_values_all_one):
    model = Model(*mock_init_values_all_one)
    sublimation_temperature = model._calculate_sublimation_temperature(1)
    sublimation_temperature_mas = model._calculate_sublimation_temperature(1*u.mas)
    assert sublimation_temperature.unit == u.K
    assert sublimation_temperature_mas.unit == u.K
    with pytest.raises(IOError):
        sublimation_temperature_K = model._calculate_sublimation_temperature(1*u.K)

def test_sublimation_radius(mock_init_values_all_one):
    model = Model(*mock_init_values_all_one)
    sublimation_radius = model._calculate_sublimation_radius(1)
    assert sublimation_radius.unit == u.mas

def test_set_grid(mock_init_values_all_one, mock_init_values, incline_params):
    # TODO: Improve this test for units and such
    # TODO: Include error tests
    model = Model(*mock_init_values_all_one)
    model_real = Model(*mock_init_values)
    grid_one_pixel = model._set_grid()
    grid_multiple_pixels = model_real._set_grid()
    grid_multiple_pixels_inclined = model_real._set_grid(incline_params)
    assert grid_one_pixel.unit == u.mas
    assert grid_multiple_pixels.unit == u.mas
    assert grid_multiple_pixels_inclined.unit == u.mas

def test_set_azimuthal_modulation(mock_init_values):
    # TODO: Improve this test for units and such
    # TODO: Include error tests

def test_temperature_gradient(mock_init_values):
    # TODO: Improve this test for units and such
    # TODO: Include error tests
    model = Model(*mock_init_values)
    inner_radius = 1
    sublimation_temperature = model._calculate_sublimation_temperature(inner_radius)
    radius = model._set_grid()
    temperature_gradient = model._temperature_gradient(radius, 0.55, inner_radius,
                                                      sublimation_temperature)
    assert temperature_gradient.unit == u.K

def test_optical_depth_gradient(mock_init_values):
    # TODO: Improve this test for units and such
    # TODO: Include error tests
    model = Model(*mock_init_values)
    radius = model._set_grid()
    optical_depth_gradient = model._optical_depth_gradient(radius, 1, 0.55, 1.)
    assert optical_depth_gradient.unit == u.dimensionless_unscaled

def test_flux_per_pixel(mock_init_values_all_one):
    model = Model(*mock_init_values_all_one)


if __name__ == "__main__":
    ...
