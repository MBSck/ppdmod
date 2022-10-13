import pytest

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
    image_size = u.Quantity(128, unit=u.dimensionless_unscaled, dtype=int)
    return [50*u.mas, image_size, 1500*u.K,
            7900*u.K, 140*u.pc, 19*c.L_sun, image_size]

@pytest.fixture
def mock_init_values_all_one():
    image_size = u.Quantity(1, unit=u.dimensionless_unscaled, dtype=int)
    return [1*u.mas, image_size, 1*u.K,
            1*u.K, 1*u.pc, 1*c.L_sun, image_size]

@pytest.fixture
def mock_wavelength():
    return 8*u.um

@pytest.fixture
def mock_incline_params():
    return [0.6, 180]

@pytest.fixture
def mock_modulation_params():
    return [0.5, 180]

################################ MODEL - TESTS ###########################################

# NOTE: All tests are written for inputs in astropy.units.Quantities
# TODO: Write tests for setters and getters

def test_init_model(mock_init_values):
    # TODO: Maybe extend this text
    assert Model(*mock_init_values)

def test_image_centre_property():
    ...

def test_pixel_scaling_property():
    ...

# TODO: Improve this test for units and such
# TODO: Also for the inclination
# TODO: Include error tests
# TODO: Make tests for borders of inputs
def test_set_grid(mock_init_values_all_one, mock_init_values, mock_incline_params):
    model = Model(*mock_init_values_all_one)
    model_real = Model(*mock_init_values)
    grid_one_pixel = model._set_grid()
    grid_multiple_pixels = model_real._set_grid()
    grid_multiple_pixels_inclined = model_real._set_grid(mock_incline_params)
    assert grid_one_pixel.unit == u.mas
    assert grid_multiple_pixels.unit == u.mas
    assert grid_multiple_pixels_inclined.unit == u.mas

# Implement this test
def test_set_uv_grid():
    ...

# TODO: Improve this test for units and such
# TODO: Include error tests
def test_set_azimuthal_modulation(mock_init_values, mock_modulation_params):
    model = Model(*mock_init_values)
    grid = model._set_grid()
    modulated_grid = model._set_azimuthal_modulation(grid, *mock_modulation_params)
    assert modulated_grid.unit == grid.unit
    assert modulated_grid.value.shape == grid.value.shape

# TODO: Implement this test
def test_set_all_zeros():
    ...

# TODO: Implement this test
def test_set_all_ones():
    ...

# NOTE: Non implemented in the base class
def test_eval_model():
    ...

# TODO: Implement this test
def test_eval_object():
    ...

# NOTE: Non implemented in the base class
def test_eval_flux():
    ...

