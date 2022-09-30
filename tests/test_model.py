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

@pytest.fixture
def mock_init_values_all_one():
    return [1, 1, 1, 1, 1, 1]

@pytest.fixture
def mock_values_hd142666():
    low_bound = [50, 128, 1500, 7900, 120, 13]
    middle = [50, 128, 1500, 7900, 140, 19]
    high_bound = [50, 128, 1500, 7900, 160, 25]
    return low_bound, middle, high_bound


################################ ReadoutFits - TESTS #####################################

def test_init(mock_init_values):
    assert Model(*mock_init_values)

def test_calculate_orbital_radius_from_parallax(mock_init_values_all_one):
    # TODO: Make test work for the other way round
    model = Model(*mock_init_values_all_one)
    orbital_radius = model._calculate_orbital_radius_from_parallax(1*u.arcsec)
    model.distance = 100
    orbital_radius_hundred_pc = model._calculate_orbital_radius_from_parallax(1*u.arcsec)
    assert orbital_radius.unit == u.m
    assert orbital_radius.value == c.au.value
    # FIXME: Is this test correct?
    assert (orbital_radius_hundred_pc.value < 100*c.au.value)\
        and (orbital_radius_hundred_pc.value > (100*c.au.value) - 1e6)

def test_calculate_stellar_radius(mock_values_hd142666):
    # model_low, model_middle, model_high = [Model(values)\
                                           # for values in mock_values_hd142666]
    ...

def test_calbulate_sublimation_temperature(mock_init_values):
    ...

def test_calculate_sublimation_radius(mock_init_values):
    ...


if __name__ == "__main__":
    ...
