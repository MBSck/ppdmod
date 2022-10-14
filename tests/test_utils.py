import pytest

import numpy as np
import astropy.units as u
import astropy.constants as c

from ppdmod.functionality.model import Model
from ppdmod.functionality.utils import _convert_orbital_radius_to_parallax,\
    _convert_parallax_to_orbital_radius, _calculate_sublimation_radius,\
    _calculate_sublimation_temperature, _calculate_stellar_radius, temperature_gradient,\
    optical_depth_gradient, _set_params_from_priors, _set_units_for_priors, flux_per_pixel,\
    _make_priors, _make_params_from_priors, _make_params, _make_component,\
    make_fixed_params, make_disc_params, make_ring_component, make_delta_component,\
    _check_attributes, check_and_convert, stellar_flux, IterNamespace


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
def mock_wavelength():
    return 8*u.um

@pytest.fixture
def mock_pixel_scaling():
    return 30*u.mas/128*u.dimensionless_unscaled

@pytest.fixture
def mock_ab_aur():
    return [10000*u.K, 139*u.pc, 60*c.L_sun]

@pytest.fixture
def mock_sublimation_radius_input():
    return [1500*u.K, 140*u.pc, 19*c.L_sun]

@pytest.fixture
def mock_sublimation_temp_input():
    return [1500*u.mas, 140*u.pc, 19*c.L_sun]

@pytest.fixture
def mock_temp_grad_input(mock_init_values):
    model = Model(*mock_init_values)
    radius = model._set_grid()
    return [radius, 0.55*u.dimensionless_unscaled, 1*u.mas, 1500*u.K]

@pytest.fixture
def mock_optical_depth_grad_input(mock_init_values):
    model = Model(*mock_init_values)
    radius = model._set_grid()
    return [radius, 0.55*u.dimensionless_unscaled, 1.*u.mas, 1.*u.dimensionless_unscaled]

@pytest.fixture
def mock_temp_gradient(mock_temp_grad_input):
    return temperature_gradient(*mock_temp_grad_input)

@pytest.fixture
def mock_optical_depth_gradient(mock_optical_depth_grad_input):
    return optical_depth_gradient(*mock_optical_depth_grad_input)

################################ UTILS: PHYSICS- TESTS ###################################

def test_IterNamespace(mock_labels, mock_params):
    mock_dict = dict(zip(mock_labels, mock_params))
    mock_namespace = IterNamespace(**mock_dict)
    assert mock_namespace._fields == mock_labels
    assert tuple(value for value in mock_namespace) == mock_params
    assert mock_namespace.axis_ratio == mock_params[0]
    assert mock_namespace.pa == mock_params[1]

def test_convert_orbital_radius_to_parallax():
    orbital_radius = _convert_orbital_radius_to_parallax(1*u.au, 1*u.pc)
    orbital_radius_hundred_pc = _convert_orbital_radius_to_parallax(1*u.au, 100*u.pc)
    assert orbital_radius.unit == u.mas
    # These values have been adapted for u.mas instead of u.arcsec
    assert (orbital_radius.value <= 1e3)\
        and (orbital_radius.value > (1e3 - 0.1))
    assert (orbital_radius_hundred_pc.value < 0.01e3)\
        and (orbital_radius_hundred_pc.value > (0.01e3 - 0.1))

def test_convert_parallax_to_orbital_radius():
    orbital_radius = _convert_parallax_to_orbital_radius(1e3*u.mas, 1*u.pc)
    orbital_radius_hundred_pc = _convert_parallax_to_orbital_radius(1e3*u.mas, 100*u.pc)
    assert orbital_radius.unit == u.m
    assert orbital_radius.value == c.au.value
    # FIXME: Is this test correct?
    assert (orbital_radius_hundred_pc.value <= (1e2*c.au.value + 0.1))\
        and (orbital_radius_hundred_pc.value > (1e2*c.au.value - 1e6))

def test_calculate_stellar_radius(mock_ab_aur):
    # TODO: Check if the radius calculation is accurate enough, AB Aur etc.
    # TODO: Implement more unit tests
    effective_temp, distance, lum_star = mock_ab_aur
    stellar_radius_ab_aur = _calculate_stellar_radius(lum_star,
                                                      effective_temp).to(u.R_sun)
    assert stellar_radius_ab_aur.unit == u.R_sun
    assert (stellar_radius_ab_aur.value <= (2.4+0.2))\
        and (stellar_radius_ab_aur.value >= (2.4-0.2))

def test_stellar_flux(mock_ab_aur, mock_wavelength):
    # TODO: Write testes that check the flux value against values of real stars
    flux = stellar_flux(mock_wavelength, *mock_ab_aur)
    assert flux.unit == u.Jy

def test_calculate_sublimation_radius(mock_sublimation_radius_input):
    wrong_temperature_input = [10*u.mas, 140*u.pc, 19*u.K]
    wrong_distance_input = [10*u.mas, 140*u.m, 19*c.L_sun]
    wrong_luminosity_input = [10*u.mas, 140*u.pc, 19*u.K]
    sublimation_radius = _calculate_sublimation_radius(*mock_sublimation_radius_input)
    assert sublimation_radius.unit == u.mas
    with pytest.raises(IOError):
        _calculate_sublimation_radius(*wrong_temperature_input)
    with pytest.raises(IOError):
        _calculate_sublimation_radius(*wrong_distance_input)
    with pytest.raises(IOError):
        _calculate_sublimation_radius(*wrong_luminosity_input)

def test_calculate_sublimation_temperature(mock_sublimation_temp_input):
    wrong_radius_input = [1500*u.K, 140*u.pc, 19*u.K]
    wrong_distance_input = [10*u.mas, 140*u.m, 19*c.L_sun]
    wrong_luminosity_input = [10*u.mas, 140*u.pc, 19*u.K]
    sublimation_temperature =\
        _calculate_sublimation_temperature(*mock_sublimation_temp_input)
    assert sublimation_temperature.unit == u.K
    with pytest.raises(IOError):
        _calculate_sublimation_temperature(*wrong_radius_input)
    with pytest.raises(IOError):
        _calculate_sublimation_temperature(*wrong_distance_input)
    with pytest.raises(IOError):
        _calculate_sublimation_temperature(*wrong_luminosity_input)

# TODO: Test for real values
def test_temperature_gradient(mock_temp_grad_input):
    wrong_radius_input = [5*u.K, 0.55, 1*u.mas, 1500*u.K]
    wrong_power_law_input = [1*u.mas, 0.55*u.m, 1*u.mas, 1500*u.K]
    wrong_inner_radius_input = [1*u.mas, 0.55, 1*u.K, 1500*u.K]
    wrong_sub_temp_input = [1*u.mas, 0.55, 1*u.mas, 1*u.mas]
    temperature = temperature_gradient(*mock_temp_grad_input)
    assert temperature.unit == u.K
    with pytest.raises(IOError):
        temperature_gradient(*wrong_radius_input)
    with pytest.raises(IOError):
        temperature_gradient(*wrong_power_law_input)
    with pytest.raises(IOError):
        temperature_gradient(*wrong_inner_radius_input)
    with pytest.raises(IOError):
        temperature_gradient(*wrong_sub_temp_input)

# TODO: Test for real values
def test_optical_depth_gradient(mock_optical_depth_grad_input):
    wrong_radius_input = [5*u.K, 0.55, 1*u.mas, 1*u.dimensionless_unscaled]
    wrong_power_law_input = [1*u.mas, 0.55*u.m, 1*u.mas, 1*u.dimensionless_unscaled]
    wrong_inner_radius_input = [1*u.mas, 0.55, 1*u.K, 1*u.dimensionless_unscaled]
    wrong_inner_optical_depth_input = [1*u.mas, 0.55, 1*u.mas, 1*u.K]
    optical_depth = optical_depth_gradient(*mock_optical_depth_grad_input)
    assert optical_depth.unit == u.dimensionless_unscaled
    with pytest.raises(IOError):
        optical_depth_gradient(*wrong_radius_input)
    with pytest.raises(IOError):
        optical_depth_gradient(*wrong_power_law_input)
    with pytest.raises(IOError):
        optical_depth_gradient(*wrong_inner_radius_input)
    with pytest.raises(IOError):
        optical_depth_gradient(*wrong_inner_optical_depth_input)

def test_flux_per_pixel(mock_wavelength, mock_temp_gradient,
                        mock_optical_depth_gradient, mock_pixel_scaling):
    flux_px = flux_per_pixel(mock_wavelength, mock_temp_gradient,
                             mock_optical_depth_gradient, mock_pixel_scaling)
    wrong_wavelength_input = [1*u.mas, mock_temp_gradient,
                              mock_optical_depth_gradient, mock_pixel_scaling]
    wrong_temperature_input = [mock_wavelength, 1*u.mas,
                               mock_optical_depth_gradient, mock_pixel_scaling]
    wrong_inner_optical_depth_input = [mock_wavelength, mock_temp_gradient,
                                       1*u.K, mock_pixel_scaling]
    wrong_pixel_scaling_input = [mock_wavelength, mock_temp_gradient,
                                 mock_optical_depth_gradient, 1*u.m]
    assert flux_px.value.shape == mock_temp_gradient.value.shape
    assert flux_px.unit == u.Jy
    with pytest.raises(IOError):
        flux_per_pixel(*wrong_wavelength_input)
    with pytest.raises(IOError):
        flux_per_pixel(*wrong_temperature_input)
    with pytest.raises(IOError):
        flux_per_pixel(*wrong_inner_optical_depth_input)
    with pytest.raises(IOError):
        flux_per_pixel(*wrong_pixel_scaling_input)

################################### Fixtures #############################################

@pytest.fixture
def mock_priors():
    return [0., 1.], [0, 180]

@pytest.fixture
def mock_prior_units():
    return u.dimensionless_unscaled, u.deg

@pytest.fixture
def mock_params():
    return 10*u.dimensionless_unscaled, 5*u.deg

@pytest.fixture
def mock_wrong_params():
    return 10, 5*u.deg

@pytest.fixture
def mock_wrong_unit_params():
    return 10*u.mas, 5*u.deg

@pytest.fixture
def mock_mod_priors():
    return ([0, 180], [0., 2.])

@pytest.fixture
def mock_mod_params():
    return 10*u.deg, 5*u.dimensionless_unscaled

@pytest.fixture
def mock_labels():
    return "axis_ratio", "pa"

@pytest.fixture
def mock_wrong_labels():
    return "flux1", "pa"

@pytest.fixture
def mock_fixed_params():
    return [50, 128, 1500, 7900, 140, 19, 1]

@pytest.fixture
def mock_disc_priors():
    return [[0., 1.], [0., 1.]]

@pytest.fixture
def attributes():
    return "axis_ratio", "pa"

@pytest.fixture
def units():
    return u.dimensionless_unscaled, u.deg

@pytest.fixture
def component_names():
    return "Ring", "Gauss"

################################ UTILS: TOOLS - TESTS ####################################

def test_set_units_for_priors(mock_priors, mock_prior_units):
    priors = _set_units_for_priors(mock_priors, mock_prior_units)
    assert all(prior.unit == mock_prior_units[i] for i, prior in enumerate(priors))
    assert all([isinstance(prior, np.ndarray) for prior in priors])
    assert all((prior.value[0] == mock_priors[i][0])\
               and (prior.value[1] == mock_priors[i][1])\
               for i, prior in enumerate(priors))

def test_set_params_from_priors(mock_priors, mock_prior_units):
    priors = _set_units_for_priors(mock_priors, mock_prior_units)
    params = _set_params_from_priors(priors)
    assert all(prior.unit == params[i].unit for i, prior in enumerate(priors))
    for i, prior in enumerate(priors):
        assert params[i] > prior[0]
        assert params[i] < prior[1]
    assert all((prior[0] <= params[i])\
               and (prior[1] >= params[i])\
               for i, prior in enumerate(priors))
    assert all(isinstance(param.value, float) for param in params)

def test_make_params(mock_params, mock_wrong_params, mock_labels):
    params = _make_params(mock_params, mock_labels)
    with pytest.raises(IOError):
        _make_params(mock_wrong_params, mock_labels)
    assert params.axis_ratio.unit == u.dimensionless_unscaled
    assert params.pa.unit == u.deg
    assert params.pa.value == 5
    assert params.axis_ratio.value == 10


# NOTE: Combination of two tested functions -> No test necessary
def test_make_priors():
    ...


# NOTE: Combination of two tested functions -> No test necessary
def test_make_params_from_priors():
    ...


def test_make_component(component_names, mock_params, mock_priors, mock_labels,
                        mock_prior_units, mock_mod_priors, mock_mod_params):
    ring_name, gauss_name = component_names
    ring_from_priors = _make_component("ring", ring_name, mock_priors,
                                      mock_labels, mock_prior_units)
    gauss_from_priors = _make_component("gauss", gauss_name, mock_priors,
                                       mock_labels, mock_prior_units)
    ring_params = _make_component("ring", ring_name, mock_priors,
                                 mock_labels, mock_prior_units, params=mock_params)
    ring_mod_priors = _make_component("ring", ring_name, mock_priors,
                                     mock_labels, mock_prior_units,
                                     mod_priors=mock_mod_priors)
    ring_mod_params = _make_component("ring", ring_name, mock_priors,
                                     mock_labels, mock_prior_units,
                                     mod_params=mock_mod_params)
    assert isinstance(ring_from_priors.params.pa, u.Quantity)
    assert isinstance(gauss_from_priors.params.pa, u.Quantity)
    assert isinstance(ring_params.params.pa, u.Quantity)
    assert isinstance(ring_mod_priors.mod_params.mod_angle, u.Quantity)
    assert isinstance(ring_mod_params.mod_params.mod_amp, u.Quantity)


def test_check_attributes(attributes, units, mock_priors, mock_params, mock_labels,
                          mock_prior_units, mock_wrong_labels, mock_wrong_unit_params,
                          component_names):
    # TODO: Make the errors more specific
    ring_name, _ = component_names
    ring_params = _make_component("ring", ring_name, mock_priors, mock_labels,
                                 mock_prior_units, params=mock_params)
    ring_params_wrong_labels = _make_component("ring", ring_name, mock_priors,
                                              mock_wrong_labels, mock_prior_units,
                                              params=mock_params)
    ring_params_wrong_units = _make_component("ring", ring_name, mock_priors,
                                             mock_labels, mock_prior_units,
                                             params=mock_wrong_unit_params)
    with pytest.raises(IOError):
        _check_attributes(ring_params_wrong_labels.params, attributes, units)
    with pytest.raises(IOError):
        _check_attributes(ring_params_wrong_units.params, attributes, units)
    assert _check_attributes(ring_params.params, attributes, units)


def test_check_and_convert(mock_priors, mock_prior_units,
                           mock_labels, mock_wrong_labels, mock_params):
    priors = _make_priors(mock_priors, mock_prior_units, mock_labels)
    params = _make_params_from_priors(priors, mock_labels)
    params_IterNamespace = check_and_convert(params, mock_labels, mock_prior_units)
    params_converted = check_and_convert(mock_params, mock_labels, mock_prior_units)
    assert params == params_IterNamespace
    assert isinstance(params_converted, IterNamespace)
    with pytest.raises(IOError):
        check_and_convert(params, mock_wrong_labels, mock_prior_units)

# TODO: Implementation test
def test_make_disc_params(mock_disc_priors):
    units = [u.dimensionless_unscaled, u.dimensionless_unscaled]
    disc_params = make_disc_params(*mock_disc_priors)
    assert all(param.unit == units[i] for i, param in enumerate(disc_params.params))
    assert len(disc_params) == len(units)

# TODO: Implementation test
def test_make_fixed_params(mock_fixed_params):
    units = [u.mas, u.dimensionless_unscaled, u.K,
             u.K, u.pc, u.W, u.dimensionless_unscaled, u.dimensionless_unscaled]
    wrong_fov = [50*u.m, 128, 1500, 7900, 140, 19, 1]
    wrong_image_size = [50, 128*u.K, 1500, 7900, 140, 19, 1]
    wrong_temp = [50, 128, 1500*u.mas, 7900, 140, 19, 1]
    wrong_eff_temp = [50, 128, 1500, 7900*u.mas, 140, 19, 1]
    wrong_distance = [50, 128, 1500, 7900, 140*u.m, 19, 1]
    wrong_lum = [50, 128, 1500, 7900, 140, 19*u.K, 1]
    wrong_tau = [50, 128, 1500, 7900, 140, 19, 1*u.mas]
    wrong_pixel_samp = [50, 128, 1500, 7900, 140, 19, 1, 128*u.Jy]
    fixed_params = make_fixed_params(*mock_fixed_params)
    assert all(param.unit == units[i] for i, param in enumerate(fixed_params))
    assert len(fixed_params) == len(units)
    with pytest.raises(IOError):
        make_fixed_params(*wrong_fov)
    with pytest.raises(IOError):
        make_fixed_params(*wrong_image_size)
    with pytest.raises(IOError):
        make_fixed_params(*wrong_temp)
    with pytest.raises(IOError):
        make_fixed_params(*wrong_eff_temp)
    with pytest.raises(IOError):
        make_fixed_params(*wrong_distance)
    with pytest.raises(IOError):
        make_fixed_params(*wrong_lum)
    with pytest.raises(IOError):
        make_fixed_params(*wrong_tau)
    with pytest.raises(IOError):
        make_fixed_params(*wrong_pixel_samp)

# TODO: Implementation test
def test_make_ring_component():
    ...

# TODO: Implementation test
def test_make_delta_component():
    ...

