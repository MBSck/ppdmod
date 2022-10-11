import pytest

import astropy.units as u

from ppdmod.functionality.utils import make_params_tuple, make_component_tuple,\
    check_attributes

################################### Fixtures #############################################

@pytest.fixture
def mock_params():
    return 10*u.Jy, 5*u.mas

@pytest.fixture
def mock_wrong_params():
    return 10, 5*u.mas

@pytest.fixture
def mock_wrong_unit_params():
    return 10*u.mas, 5*u.mas

@pytest.fixture
def mock_labels():
    return "flux", "x"

@pytest.fixture
def mock_wrong_labels():
    return "flux1", "x"

@pytest.fixture
def attributes():
    return "flux", "x"

@pytest.fixture
def units():
    return u.Jy, u.mas

@pytest.fixture
def component_names():
    return "Ring", "Gauss"

################################ MODEL COMPONENTS - TESTS ################################

def test_make_params_tuple(mock_params, mock_wrong_params, mock_labels):
    params = make_params_tuple(mock_params, mock_labels)
    with pytest.raises(IOError):
        make_params_tuple(mock_wrong_params, mock_labels)
    assert params.x.unit == u.mas
    assert params.flux.unit == u.Jy
    assert params.flux.value == 10
    assert params.x.value == 5

def test_make_component_tuple(mock_params, mock_labels, component_names):
    # TODO: Make the errors more specific
    ring_name, gauss_name = component_names
    ring_params = make_component_tuple(ring_name, mock_params, mock_labels)
    gauss_params = make_component_tuple(gauss_name, mock_params, mock_labels)
    assert isinstance(ring_params.params.x, u.Quantity)
    assert isinstance(gauss_params.params.x, u.Quantity)

def test_check_attributes(attributes, units, mock_params, mock_labels,
                          mock_wrong_labels, mock_wrong_unit_params, component_names):
    # TODO: Make the errors more specific
    ring_name, _ = component_names
    ring_params = make_component_tuple(ring_name, mock_params, mock_labels)
    ring_params_wrong_labels = make_component_tuple(ring_name, mock_params,
                                                    mock_wrong_labels)
    ring_params_wrong_units = make_component_tuple(ring_name, mock_wrong_unit_params,
                                                   mock_labels)
    with pytest.raises(IOError):
        check_attributes(ring_params_wrong_labels.params, attributes, units)
    with pytest.raises(IOError):
        check_attributes(ring_params_wrong_units.params, attributes, units)
    assert check_attributes(ring_params.params, attributes, units)
