import pytest

import astropy.units as u
import astropy.constants as c

from ppdmod.functionality.model import Model

################################### Fixtures #############################################

@pytest.fixture
def mock_init_values():
    """The sublimation temp [astropy.units.K], the eff_temp [astropy.units.K], the
    distance of the star [astropy.units.pc], the luminosity [astropy.units.L_sun] and
    the wavelength [u.um]"""
    return [50, 128, 1500, 7900, 140, 19]

################################ COMBINED MODEL - TESTS ##################################

def test_init_combined_model():
    combined_model = CombinedModel()
    assert combined_model

