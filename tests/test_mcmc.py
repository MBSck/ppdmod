from typing import Dict
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest

from ppdmod.parameter import Parameter
from ppdmod.mcmc import chi_sq, lnprob, initiate_randomly, run_mcmc
from ppdmod.data import ReadoutFits, get_data


@pytest.fixture
def qval_file_dir() -> Path:
    """The qval-file directory."""
    return Path("data/qval")


@pytest.fixture
def fits_file() -> Path:
    """A MATISSE (.fits)-file."""
    return Path("data/fits") /\
        "hd_142666_2022-04-23T03_05_25:2022-04-23T02_28_06_AQUARIUS_FINAL_TARGET_INT.fits"


@pytest.fixture
def fit_wavelengths() -> u.um:
    """A wavelength grid."""
    return ([8.28835527e-06, 1.02322101e-05]*u.m).to(u.um)


@pytest.fixture
def wavelength_solution(fits_file: Path) -> u.um:
    """The wavelength solution of a MATISSE (.fits)-file."""
    return (ReadoutFits(fits_file).wavelength*u.m).to(u.um)


# @pytest.mark.parametrize(
#     "data, error, data_model, expected",
#     [(np.full((6,), fill_value=50), np.array([50, 60, 40, 47, 53]), 4.36)])
# def test_chi_sq(data: np.ndarray, error: np.ndarray,
#                 data_model: np.ndarray, expected: int) -> None:
#     """Tests the chi squre function."""
#     assert chi_sq(data, error, data_model) == expected


def test_lnprob() -> None:
    ...


def test_initiate_randomly() -> None:
    ...


def test_run_mcmc() -> None:
    ...
