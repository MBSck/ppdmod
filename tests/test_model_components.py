import pytest

import astropy.units as u
import astropy.constants as c

from ppdmod.model_components import DeltaComponent, RingComponent
from ppdmod.functionality.utils import _make_params

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
def wavelength():
    return 8*u.um

@pytest.fixture
def mock_params():
    params = [0.5*u.dimensionless_unscaled, 145*u.deg,
              3.*u.mas, 0.*u.mas]
    attributes = ["axis_ratio", "pa", "inner_radius", "outer_radius"]
    return _make_params(params, attributes)

################################ MODEL COMPONENTS - TESTS ################################

def test_delta_component(mock_init_values, wavelength):
    # TODO: Implement visibilities at some time
    delta = DeltaComponent(*mock_init_values)
    image = delta.eval_model()
    flux = delta.eval_flux(wavelength)
    assert image.unit == u.mas
    assert image[delta.image_centre].value == 1
    assert flux.unit == u.Jy

def test_ring_component(mock_init_values, mock_params):
    ring = RingComponent(*mock_init_values)
    image = ring.eval_model(mock_params)
    assert image.unit == u.mas

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
