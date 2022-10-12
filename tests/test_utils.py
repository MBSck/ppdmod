import pytest

import numpy as np
import astropy.units as u

from ppdmod.functionality.utils import make_params_tuple, make_component_tuple,\
    check_attributes, _set_units_for_priors, _set_params_from_priors

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

################################ MODEL COMPONENTS - TESTS ################################

def test_make_params_tuple(mock_params, mock_wrong_params, mock_labels):
    params = make_params_tuple(mock_params, mock_labels)
    with pytest.raises(IOError):
        make_params_tuple(mock_wrong_params, mock_labels)
    assert params.axis_ratio.unit == u.dimensionless_unscaled
    assert params.pa.unit == u.deg
    assert params.pa.value == 5
    assert params.axis_ratio.value == 10


def test_make_component_tuple(mock_params, mock_priors, mock_labels,
                              mock_prior_units, component_names):
    # TODO: Make the errors more specific
    ring_name, gauss_name = component_names
    ring_from_priors = make_component_tuple(ring_name, mock_priors,
                                            mock_labels, mock_prior_units)
    gauss_from_priors = make_component_tuple(gauss_name, mock_priors,
                                             mock_labels, mock_prior_units)
    ring = make_component_tuple(ring_name, mock_priors,
                                mock_labels, mock_prior_units, mock_params)
    gauss = make_component_tuple(gauss_name, mock_priors,
                                 mock_labels, mock_prior_units, mock_params)
    assert isinstance(ring_from_priors.params.pa, u.Quantity)
    assert isinstance(gauss_from_priors.params.pa, u.Quantity)
    assert isinstance(ring.params.pa, u.Quantity)
    assert isinstance(gauss.params.pa, u.Quantity)


def test_check_attributes(attributes, units, mock_priors, mock_params, mock_labels,
                          mock_prior_units, mock_wrong_labels, mock_wrong_unit_params,
                          component_names):
    # TODO: Make the errors more specific
    ring_name, _ = component_names
    ring_params = make_component_tuple(ring_name, mock_priors, mock_labels,
                                       mock_prior_units, mock_params)
    ring_params_wrong_labels = make_component_tuple(ring_name, mock_priors,
                                                    mock_wrong_labels, mock_prior_units,
                                                    mock_params)
    ring_params_wrong_units = make_component_tuple(ring_name, mock_priors,
                                                    mock_labels, mock_prior_units,
                                                    mock_wrong_unit_params)
    with pytest.raises(IOError):
        check_attributes(ring_params_wrong_labels.params, attributes, units)
    with pytest.raises(IOError):
        check_attributes(ring_params_wrong_units.params, attributes, units)
    assert check_attributes(ring_params.params, attributes, units)


def test_set_units_for_priors(mock_priors, mock_prior_units):
    priors = _set_units_for_priors(mock_priors, mock_prior_units)
    assert priors[1].unit == u.deg
    assert isinstance(priors, list)
    assert all(isinstance(prior, u.Quantity) for prior in priors)
    assert all(isinstance(prior.value, np.ndarray) for prior in priors)


def test_set_params_from_priors(mock_priors, mock_labels, mock_prior_units):
    priors = _set_units_for_priors(mock_priors, mock_prior_units)
    priors = make_params_tuple(priors, mock_labels)
    params = _set_params_from_priors(priors)
    assert all(isinstance(param, u.Quantity) for param in params)
    assert all(isinstance(param.value, np.ndarray) for param in params)

