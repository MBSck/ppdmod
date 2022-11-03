import pytest
import astropy.units as u

import ppdmod.lib.utils as utils

from ppdmod.lib.model import Model

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
    model = Model(mock_fixed_params_namespace)
    assert model
    assert model.fixed_params == mock_fixed_params_namespace
    assert model.pixel_scaling ==\
        mock_fixed_params_namespace.fov/mock_fixed_params_namespace.pixel_sampling
    assert model._component_name == None
    assert model._polar_angle == None

# TODO: Implement test
def test_image_centre_property(mock_fixed_params_namespace):
    model_delta = Model(mock_fixed_params_namespace)
    model = Model(mock_fixed_params_namespace)
    model_delta._component_name = "delta"
    model_delta.fixed_params.pixel_sampling = 1024
    model_delta.fixed_params.pixel_sampling = 1024
    assert model_delta.image_centre == (64, 64)
    assert model.image_centre == (512, 512)

# TODO: Improve this test for units and such
# TODO: Also for the inclination
# TODO: Make tests for borders of inputs
def test_set_grid(mock_fixed_params_namespace, mock_incline_params):
    model = Model(mock_fixed_params_namespace)
    model_delta = Model(mock_fixed_params_namespace)
    model_delta._component_name = "delta"
    grid = model._set_grid()
    grid_inclined = model._set_grid(mock_incline_params)
    assert grid.unit == u.mas
    assert grid_inclined.unit == u.mas
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

