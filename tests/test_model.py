import pytest
import astropy.units as u

import ppdmod.functionality.utils as utils

from ppdmod.functionality.model import Model

# TODO: Improve a lot of theses tests and check how to test properly

################################### Fixtures #############################################

@pytest.fixture
def mock_fixed_params_namespace():
    """The sublimation temp [astropy.units.K], the eff_temp [astropy.units.K], the
    distance of the star [astropy.units.pc], the luminosity [astropy.units.L_sun] and
    the wavelength [u.um]"""
    params = [50, 128, 1500, 7900, 140, 19]
    return utils.make_fixed_params(*params)

@pytest.fixture
def mock_incline_params():
    return [0.6*u.dimensionless_unscaled, 180*u.deg]

################################ MODEL - TESTS ###########################################

def test_init_model(mock_fixed_params_namespace):
    # TODO: Maybe extend this text
    assert Model(mock_fixed_params_namespace)

# TODO: Implement test
def test_image_centre_property():
    ...

# TODO: Improve this test for units and such
# TODO: Also for the inclination
# TODO: Make tests for borders of inputs
def test_set_grid(mock_fixed_params_namespace, mock_incline_params):
    model = Model(mock_fixed_params_namespace)
    grid_multiple_pixels = model._set_grid()
    grid_multiple_pixels_inclined = model._set_grid(mock_incline_params)
    assert grid_multiple_pixels.unit == u.mas
    assert grid_multiple_pixels_inclined.unit == u.mas

    model.fixed_params.image_size = u.Quantity(1, unit=u.dimensionless_unscaled,
                                                    dtype=int)
    model.fixed_params.pixel_sampling = u.Quantity(1, unit=u.dimensionless_unscaled,
                                                    dtype=int)
    grid_one_pixel = model._set_grid()
    assert grid_one_pixel.unit == u.mas

# TODO: Improve this test for units and such
def test_set_azimuthal_modulation(mock_fixed_params_namespace, mock_incline_params):
    model = Model(mock_fixed_params_namespace)
    grid = model._set_grid()
    modulated_grid = model._set_azimuthal_modulation(grid, *mock_incline_params)
    assert modulated_grid.unit == grid.unit
    assert modulated_grid.value.shape == grid.value.shape

