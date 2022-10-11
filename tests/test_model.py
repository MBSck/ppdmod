import pytest

import astropy.units as u
import astropy.constants as c

from ppdmod.functionality.model import CombinedModel, Model

# TODO: Improve a lot of theses tests and check how to test properly

################################### Fixtures #############################################

@pytest.fixture
def mock_init_values():
    """The sublimation temp [astropy.units.K], the eff_temp [astropy.units.K], the
    distance of the star [astropy.units.pc], the luminosity [astropy.units.L_sun] and
    the wavelength [u.um]"""
    return [50, 128, 1500, 7900, 140, 19]

@pytest.fixture
def mock_values_ab_aur():
    """These values can be used to compare the stellar flux and radius calculations"""
    return [50, 128, 1500, 10000, 139, 60]

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

@pytest.fixture
def temperature_gradient(mock_init_values):
    # TODO: Improve this test for units and such
    # TODO: Include error tests
    model = Model(*mock_init_values)
    inner_radius = 1
    sublimation_temperature = model._calculate_sublimation_temperature(inner_radius)
    radius = model._set_grid()
    temperature_gradient = model._temperature_gradient(radius, 0.55, inner_radius,
                                                      sublimation_temperature)
    return temperature_gradient

@pytest.fixture
def optical_depth_gradient(mock_init_values):
    # TODO: Improve this test for units and such
    # TODO: Include error tests
    model = Model(*mock_init_values)
    radius = model._set_grid()
    optical_depth_gradient = model._optical_depth_gradient(radius, 1, 0.55, 1.)
    return optical_depth_gradient

################################ MODEL - TESTS ###########################################

# TODO: Write tests for setters and getters

def test_init_model(mock_init_values):
    # TODO: Maybe extend this text
    assert Model(*mock_init_values)

def test_field_of_view_property(mock_init_values):
    ...

def test_image_size_property(mock_init_values):
    ...

def test_pixel_sampling_property(mock_init_values):
    ...

def test_sublimation_temperature_property(mock_init_values):
    ...

def test_effective_temperature_property(mock_init_values):
    ...

def test_luminosity_property(mock_init_values):
    ...

def test_distance_property(mock_init_values):
    ...

def test_convert_parallax_to_orbital_radius(mock_init_values_all_one):
    # TODO: Make test work for the other way round
    model = Model(*mock_init_values_all_one)
    orbital_radius = model._convert_parallax_to_orbital_radius(1e3*u.mas)
    model.distance = 100
    orbital_radius_hundred_pc = model._convert_parallax_to_orbital_radius(1e3*u.mas)
    assert orbital_radius.unit == u.m
    assert orbital_radius.value == c.au.value
    # FIXME: Is this test correct?
    assert (orbital_radius_hundred_pc.value <= (1e2*c.au.value + 0.1))\
        and (orbital_radius_hundred_pc.value > (1e2*c.au.value - 1e6))

def test_convert_orbital_radius_to_parallax(mock_init_values_all_one):
    # TODO: Make test work for the other way round
    model = Model(*mock_init_values_all_one)
    orbital_radius = model._convert_orbital_radius_to_parallax(1*u.au)
    model.distance = 100
    orbital_radius_hundred_pc = model._convert_orbital_radius_to_parallax(1*u.au)
    assert orbital_radius.unit == u.mas
    # These values have been adapted for u.mas instead of u.arcsec
    assert (orbital_radius.value <= 1e3)\
        and (orbital_radius.value > (1e3 - 0.1))
    assert (orbital_radius_hundred_pc.value < 0.01e3)\
        and (orbital_radius_hundred_pc.value > (0.01e3 - 0.1))

def test_calculate_stellar_radius(mock_values_ab_aur):
    # TODO: Check if the radius calculation is accurate enough, AB Aur etc.
    # TODO: Implement more unit tests
    model = Model(*mock_values_ab_aur)
    stellar_radius_ab_aur = model._calculate_stellar_radius().to(u.R_sun)
    assert stellar_radius_ab_aur.unit == u.R_sun
    assert (stellar_radius_ab_aur.value <= (2.4+0.2))\
        and (stellar_radius_ab_aur.value >= (2.4-0.2))

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

def test_set_uv_grid(mock_init_values_all_one):
    # Implement this test
    ...

def test_set_azimuthal_modulation(mock_init_values):
    # TODO: Improve this test for units and such
    # TODO: Include error tests
    model = Model(*mock_init_values)
    grid = model._set_grid()
    modulated_grid = model._set_azimuthal_modulation(grid, 150, 1)
    assert modulated_grid.unit == grid.unit
    assert modulated_grid.value.shape == grid.value.shape

def test_set_all_zeros(mock_init_values):
    ...

def test_set_all_ones(mock_init_values):
    ...

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

def test_flux_per_pixel(mock_init_values_all_one, wavelength,
                        temperature_gradient, optical_depth_gradient):
    model = Model(*mock_init_values_all_one)
    flux_per_pixel = model._flux_per_pixel(wavelength, temperature_gradient,
                                           optical_depth_gradient)
    assert flux_per_pixel.value.shape == temperature_gradient.value.shape
    assert flux_per_pixel.unit == u.Jy

def test_stellar_flux(mock_values_ab_aur, wavelength):
    # TODO: Write testes that check the flux value against values of real stars
    model = Model(*mock_values_ab_aur)
    stellar_flux = model._stellar_flux(wavelength)
    assert stellar_flux.unit == u.Jy

################################ COMBINED MODEL - TESTS ##################################

def test_init_combined_model():
    combined_model = CombinedModel()
    assert combined_model

if __name__ == "__main__":
    ...
