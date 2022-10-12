import pytest

import astropy.units as u
import astropy.constants as c

from ppdmod.model_components.delta import Delta

################################### Fixtures #############################################

@pytest.fixture
def mock_init_values():
    """The sublimation temp [astropy.units.K], the eff_temp [astropy.units.K], the
    distance of the star [astropy.units.pc], the luminosity [astropy.units.L_sun] and
    the wavelength [u.um]"""
    return [50, 128, 1500, 7900, 140, 19]

################################ MODEL COMPONENTS - TESTS ################################

def test_delta_component(mock_init_values):
    # TODO: Implement visibilities at some time
    delta = Delta(*mock_init_values)
    image = delta.eval_model()
    # visibilities = delta.eval_vis()
    assert image.unit == u.mas
    assert image[delta.image_centre[0], delta.image_centre[0]] == 1*u.mas

def test_ring_component():
    ...

def test_uniform_disk_component():
    ...

def test_inclined_disk_component():
    ...

def test_gauss_component():
    ...

def test_optically_thin_sphere_component():
    ...

def test_binary_component():
    ...
