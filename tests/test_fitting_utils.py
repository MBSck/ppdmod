import pytest
import numpy as np
import astropy.units as u

import ppdmod.lib.fitting_utils as fit_tools

@pytest.fixture
def mock_flux_data():
    flux = [np.random.randint(0, 4, 12), np.random.randint(0, 4, 12)]*u.Jy
    fluxerr = flux.copy()*0.1
    return flux, fluxerr

@pytest.fixture
def mock_model_data():
    return [np.random.randint(0, 4, 12), np.random.randint(0, 4, 12)]*u.Jy

@pytest.fixture
def mock_lnf():
    return -0.578


def test_chi_sq(mock_flux_data, mock_model_data, mock_lnf):
    chi_sq = fit_tools.chi_sq(*mock_flux_data, mock_model_data, mock_lnf)
