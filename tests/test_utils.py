import pytest
import numpy as np
import astropy.units as u
import astropy.constants as c

import ppdmod.lib.utils as utils

from ppdmod.lib.model import Model


################################### Fixtures #############################################

@pytest.fixture
def mock_fixed_params_namespace(mock_fixed_params):
    """The sublimation temp [astropy.units.K], the eff_temp [astropy.units.K], the
    distance of the star [astropy.units.pc], the luminosity [astropy.units.L_sun] and
    the wavelength [u.um]"""
    return utils.make_fixed_params(*mock_fixed_params)

@pytest.fixture
def mock_wavelength():
    return 8*u.um

@pytest.fixture
def mock_pixel_scaling():
    return 30*u.mas/u.Quantity(128, unit=u.dimensionless_unscaled, dtype=int)

@pytest.fixture
def mock_ab_aur():
    return [10000*u.K, 139*u.pc, 60*c.L_sun]

@pytest.fixture
def mock_temp_grad_input(mock_fixed_params_namespace):
    model = Model(mock_fixed_params_namespace)
    radius = model._set_grid()
    return [radius, 0.55*u.dimensionless_unscaled, 1*u.mas, 1500*u.K]

@pytest.fixture
def mock_optical_depth_grad_input(mock_fixed_params_namespace):
    model = Model(mock_fixed_params_namespace)
    radius = model._set_grid()
    return [radius, 0.55*u.dimensionless_unscaled, 1.*u.mas, 1.*u.dimensionless_unscaled]

@pytest.fixture
def mock_temp_gradient(mock_temp_grad_input):
    return utils.temperature_gradient(*mock_temp_grad_input)

@pytest.fixture
def mock_optical_depth_gradient(mock_optical_depth_grad_input):
    return utils.optical_depth_gradient(*mock_optical_depth_grad_input)

################################ UTILS: PHYSICS- TESTS ###################################

def test_convert_orbital_radius_to_parallax():
    orbital_radius_one = utils._convert_orbital_radius_to_parallax(1*u.au, 1*u.pc)
    orbital_radius_hundred_pc = utils._convert_orbital_radius_to_parallax(1*u.au, 100*u.pc)
    assert orbital_radius_one.unit == u.mas
    # These values have been adapted for u.mas instead of u.arcsec
    assert np.isclose(orbital_radius_one.value, 1e3)
    assert np.isclose(orbital_radius_hundred_pc.value, 0.01e3)

def test_convert_parallax_to_orbital_radius():
    orbital_radius_one = utils._convert_parallax_to_orbital_radius(1e3*u.mas, 1*u.pc)
    orbital_radius_hundred_pc = utils._convert_parallax_to_orbital_radius(1e3*u.mas, 100*u.pc)
    assert orbital_radius_one.unit == u.m
    assert orbital_radius_one.value == c.au.value
    assert np.isclose(orbital_radius_hundred_pc.value, 1e2*c.au.value)

def test_calculate_stellar_radius(mock_ab_aur):
    # TODO: Check if the radius calculation is accurate enough, AB Aur etc.
    # TODO: Implement more tests here
    effective_temp, distance, lum_star = mock_ab_aur
    stellar_radius_ab_aur = utils._calculate_stellar_radius(lum_star,
                                                      effective_temp).to(u.R_sun)
    assert stellar_radius_ab_aur.unit == u.R_sun
    # TODO: Check if this calculation is correct
    assert (stellar_radius_ab_aur.value <= (2.4+0.2))\
        and (stellar_radius_ab_aur.value >= (2.4-0.2))

# TODO: Write testes that check the flux value against values of real stars
def test_stellar_flux(mock_ab_aur, mock_wavelength):
    flux = utils.stellar_flux(mock_wavelength, *mock_ab_aur)
    assert flux.unit == u.Jy

# TODO: Write testes that check the sublimation radius value against real values
def test_calculate_sublimation_radius(mock_fixed_params_namespace):
    fixed_params = mock_fixed_params_namespace
    sublimation_radius = utils._calculate_sublimation_radius(fixed_params.sub_temp,
                                                             fixed_params.distance,
                                                             fixed_params.lum_star)
    assert sublimation_radius.unit == u.mas

# TODO: Write testes that check the sublimation temperature value against real values
def test_calculate_sublimation_temperature(mock_fixed_params_namespace):
    fixed_params = mock_fixed_params_namespace
    inner_radius = 2.14*u.mas
    sublimation_temperature =\
        utils._calculate_sublimation_temperature(inner_radius,
                                                 fixed_params.distance,
                                                 fixed_params.lum_star)
    assert sublimation_temperature.unit == u.K

# NOTE: Astropy-function already tested there. Test only to check input
def test_temperature_gradient(mock_temp_grad_input):
    temperature = utils.temperature_gradient(*mock_temp_grad_input)
    assert temperature.unit == u.K

# NOTE: Astropy-function already tested there. Test only to check input
def test_optical_depth_gradient(mock_optical_depth_grad_input):
    optical_depth = utils.optical_depth_gradient(*mock_optical_depth_grad_input)
    assert optical_depth.unit == u.dimensionless_unscaled

def test_flux_per_pixel(mock_wavelength, mock_temp_gradient,
                        mock_optical_depth_gradient, mock_pixel_scaling):
    flux_px = utils.flux_per_pixel(mock_wavelength, mock_temp_gradient,
                                   mock_optical_depth_gradient, mock_pixel_scaling)
    assert flux_px.value.shape == mock_temp_gradient.value.shape
    assert flux_px.unit == u.Jy

################################### Fixtures #############################################

@pytest.fixture
def mock_priors():
    return [0., 1.], [0, 180]

@pytest.fixture
def mock_params():
    return 10., 5.

@pytest.fixture
def mock_mod_priors():
    return ([0, 180], [0., 2.])

@pytest.fixture
def mock_units():
    return u.dimensionless_unscaled, u.deg

@pytest.fixture
def mock_labels():
    return "axis_ratio", "pa"

@pytest.fixture
def mock_fixed_params():
    return [50, 128, 1500, 7900, 140, 19]

@pytest.fixture
def mock_disc_priors():
    return [[0., 1.], [0., 1.]]

@pytest.fixture
def component_names():
    return "Ring", "Gauss"

################################ UTILS: TOOLS - TESTS ####################################

# TODO: Add tests for all subfunctionality of this class here
def test_IterNamespace(mock_labels, mock_params):
    mock_dict = dict(zip(mock_labels, mock_params))
    mock_namespace = utils.IterNamespace(**mock_dict)
    assert mock_namespace._fields == mock_labels
    assert tuple(value for value in mock_namespace) == mock_params
    assert mock_namespace.axis_ratio == mock_params[0]
    assert mock_namespace.pa == mock_params[1]


def test_set_zeros(mock_fixed_params_namespace):
    model = Model(mock_fixed_params_namespace)
    grid = model._set_grid()
    zeros_w_unit = utils._set_zeros(grid)
    zeros_wo_unit = utils._set_zeros(grid, rvalue=True)
    assert not np.any(zeros_w_unit.value)
    assert not np.any(zeros_wo_unit)
    assert isinstance(zeros_w_unit, u.Quantity)
    assert isinstance(zeros_wo_unit, np.ndarray)


def test_set_ones(mock_fixed_params_namespace):
    model = Model(mock_fixed_params_namespace)
    grid = model._set_grid()
    zeros_w_unit = utils._set_ones(grid)
    zeros_wo_unit = utils._set_ones(grid, rvalue=True)
    assert isinstance(zeros_w_unit, u.Quantity)
    assert isinstance(zeros_wo_unit, np.ndarray)


# TODO: Implement this test
def test_rebin_image():
    ...


# TODO: Implement this test
def make_inital_guess_from_priors():
    ...


def test_make_params(mock_params, mock_units, mock_labels):
    params = utils._make_params(mock_params, mock_units, mock_labels)
    assert params.axis_ratio.unit == u.dimensionless_unscaled
    assert params.pa.unit == u.deg
    assert params.pa.value == 5
    assert params.axis_ratio.value == 10


# NOTE: Write tests for this
def test_make_priors(mock_priors, mock_units, mock_labels):
    ...


def test_make_component(component_names, mock_params,
                        mock_priors, mock_labels, mock_units):
    ring_name, gauss_name = component_names
    ring_from_priors = utils._make_component("ring", ring_name, mock_priors,
                                             mock_labels, mock_units)
    gauss_from_priors = utils._make_component("gauss", gauss_name, mock_priors,
                                              mock_labels, mock_units)
    ring_params = utils._make_component("ring", ring_name, mock_priors,
                                        mock_labels, mock_units, params=mock_params)
    ring_mod_priors = utils._make_component("ring", ring_name, mock_priors,
                                            mock_labels, mock_units,
                                            mod_priors=mock_priors)
    ring_mod_params = utils._make_component("ring", ring_name, mock_priors,
                                            mock_labels, mock_units,
                                            mod_params=mock_params)
    # TODO: Add more tests here
    assert isinstance(ring_from_priors.priors.pa, u.Quantity)
    assert isinstance(gauss_from_priors.priors.pa, u.Quantity)
    assert isinstance(ring_params.params.pa, u.Quantity)
    assert isinstance(ring_mod_priors.mod_priors.mod_angle, u.Quantity)
    assert isinstance(ring_mod_params.mod_params.mod_amp, u.Quantity)


def test_make_fixed_params(mock_fixed_params):
    units = [u.mas, u.dimensionless_unscaled, u.K,
             u.K, u.pc, u.W, u.dimensionless_unscaled]
    fixed_params = utils.make_fixed_params(*mock_fixed_params)
    assert all(param.unit == units[i] for i, param in enumerate(fixed_params))
    assert len(fixed_params) == len(units)

# TODO: Implementation test
def test_make_ring_component():
    ...

# TODO: Implementation test
def test_make_delta_component():
    ...

