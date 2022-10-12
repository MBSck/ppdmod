import pytest

import numpy as np
import astropy.units as u

from ppdmod.functionality.utils import IterNamespace, make_params, make_component,\
    check_attributes, _set_units_for_priors, _set_params_from_priors,\
    make_params_from_priors, make_priors, _check_and_convert

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
def attributes():
    return "axis_ratio", "pa"

@pytest.fixture
def units():
    return u.dimensionless_unscaled, u.deg

@pytest.fixture
def component_names():
    return "Ring", "Gauss"

################################ UTILS - TESTS ###########################################

def test_IterNamespace(mock_labels, mock_params):
    mock_dict = dict(zip(mock_labels, mock_params))
    mock_namespace = IterNamespace(**mock_dict)
    assert mock_namespace._fields == mock_labels
    assert tuple(value for value in mock_namespace) == mock_params
    assert mock_namespace.axis_ratio == mock_params[0]
    assert mock_namespace.pa == mock_params[1]

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

def test_make_params(mock_params, mock_wrong_params, mock_labels):
    params = make_params(mock_params, mock_labels)
    with pytest.raises(IOError):
        make_params(mock_wrong_params, mock_labels)
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
    ring_from_priors = make_component("ring", ring_name, mock_priors,
                                      mock_labels, mock_prior_units)
    gauss_from_priors = make_component("gauss", gauss_name, mock_priors,
                                       mock_labels, mock_prior_units)
    ring_params = make_component("ring", ring_name, mock_priors,
                                 mock_labels, mock_prior_units, params=mock_params)
    ring_mod_priors = make_component("ring", ring_name, mock_priors,
                                     mock_labels, mock_prior_units,
                                     mod_priors=mock_mod_priors)
    ring_mod_params = make_component("ring", ring_name, mock_priors,
                                     mock_labels, mock_prior_units,
                                     mod_params=mock_mod_params)
    assert isinstance(ring_from_priors.params.pa, u.Quantity)
    assert isinstance(gauss_from_priors.params.pa, u.Quantity)
    assert isinstance(ring_params.params.pa, u.Quantity)
    assert isinstance(ring_mod_priors.mod_params.mod_angle, u.Quantity)
    assert isinstance(ring_mod_params.mod_params.mod_amp, u.Quantity)
    assert not ring_from_priors.modulated
    assert ring_mod_priors.modulated
    assert ring_mod_params.modulated


def test_check_attributes(attributes, units, mock_priors, mock_params, mock_labels,
                          mock_prior_units, mock_wrong_labels, mock_wrong_unit_params,
                          component_names):
    # TODO: Make the errors more specific
    ring_name, _ = component_names
    ring_params = make_component("ring", ring_name, mock_priors, mock_labels,
                                 mock_prior_units, params=mock_params)
    ring_params_wrong_labels = make_component("ring", ring_name, mock_priors,
                                              mock_wrong_labels, mock_prior_units,
                                              params=mock_params)
    ring_params_wrong_units = make_component("ring", ring_name, mock_priors,
                                             mock_labels, mock_prior_units,
                                             params=mock_wrong_unit_params)
    with pytest.raises(IOError):
        check_attributes(ring_params_wrong_labels.params, attributes, units)
    with pytest.raises(IOError):
        check_attributes(ring_params_wrong_units.params, attributes, units)
    assert check_attributes(ring_params.params, attributes, units)


def test_check_and_convert(mock_priors, mock_prior_units,
                           mock_labels, mock_wrong_labels, mock_params):
    priors = make_priors(mock_priors, mock_prior_units, mock_labels)
    params = make_params_from_priors(priors, mock_labels)
    params_IterNamespace = _check_and_convert(params, mock_labels, mock_prior_units)
    params_converted = _check_and_convert(mock_params, mock_labels, mock_prior_units)
    assert params == params_IterNamespace
    assert isinstance(params_converted, IterNamespace)
    with pytest.raises(IOError):
        _check_and_convert(params, mock_wrong_labels, mock_prior_units)


# NOTE: Implementation of other functions
def test_make_disc_params():
    ...


# NOTE: Implementation of other functions
def test_make_fixed_params():
    ...


# NOTE: Implementation of other functions
def test_make_ring_component():
    ...


# NOTE: Implementation of other functions
def test_make_delta_component():
    ...

