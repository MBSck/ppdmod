import random
from typing import Dict, List
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest

from ppdmod import mcmc
from ppdmod.options import OPTIONS
from ppdmod.parameter import STANDARD_PARAMETERS, Parameter
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


@pytest.fixture
def shared_params() -> Dict[str, Parameter]:
    """Shared parameters."""
    return {key: Parameter(**STANDARD_PARAMETERS[key])
            for key in random.sample(list(STANDARD_PARAMETERS.keys()), 4)}


@pytest.fixture
def components_and_params() -> Dict[str, Dict]:
    """Parameters."""
    all_components = ["Star", "AsymmetricSDGreyBodyContinuum",
                      "AsymmetricSDGreyBody", "TemperatureGradient",
                      "AsymmetricSDTemperatureGradient"]
    components = random.sample(all_components, 3)
    return {component: {key: Parameter(**STANDARD_PARAMETERS[key])
            for key in random.sample(list(STANDARD_PARAMETERS.keys()), 4)}
            for component in components}


def test_set_theta_from_params(
        components_and_params: Dict[str, Dict],
        shared_params: Dict[str, Parameter]) -> None:
    """Tests the set_theta_from_params function."""
    theta = mcmc.set_theta_from_params(components_and_params,
                                       shared_params)
    len_params = sum(len(params) for params in components_and_params.values())
    assert theta.size == len_params+len(shared_params)
    assert all(theta[-len(shared_params):] ==
               [parameter.value for parameter in shared_params.values()])


# TODO: Test order of appearance for parameters.
def test_set_params_from_theta(
        components_and_params: Dict[str, Dict],
        shared_params: Dict[str, Parameter]) -> None:
    """Tests the set_params_from_theta function."""
    theta = mcmc.set_theta_from_params(components_and_params,
                                       shared_params)
    new_components_and_params, new_shared_parameters =\
        mcmc.set_params_from_theta(
            theta, components_and_params, shared_params)
    all_params = []
    for params in components_and_params.values():
        all_params.extend(list(map(lambda x: x.value, params.values())))
    new_params = []
    for params in new_components_and_params.values():
        new_params.extend(list(params.values()))
    assert all_params == new_params
    assert list(map(lambda x: x.value, shared_params.values())) ==\
        list(new_shared_parameters.values())


def test_init_randomly() -> None:
    """Tests the init_randomly function."""
    ...


# @pytest.mark.parametrize(
#     "data, error, data_model, expected",
#     [(np.full((6,), fill_value=50), np.array([50, 60, 40, 47, 53]), 4.36)])
# def test_chi_sq(data: np.ndarray, error: np.ndarray,
#                 data_model: np.ndarray, expected: int) -> None:
#     """Tests the chi squre function."""
#     assert chi_sq(data, error, data_model) == expected


def test_calculate_observables() -> None:
    """Tests the calculate_observables function."""
    ...


def test_calculate_observables_chi_sq() -> None:
    """Tests the calculate_observables chi sq function."""
    ...


@pytest.mark.parametrize(
    "values, expected", [
        ([1.5, 0.5, 0.3, 33, 0.2, 45, 1.6], 0.),
        ([1.5, 0.5, 0.3, 33, 0.2, 45, 0.6], -np.inf),
    ]
)
def test_lnprior(values: List[float], expected: float) -> None:
    """Tests the lnprior function."""
    param_names = ["rin", "p", "a", "phi",
                   "cont_weight", "pa", "elong"]
    limits = [[0, 20], [0, 1], [0, 1],
              [0, 360], [0, 1], [0, 360], [1, 50]]
    params = {name: Parameter(**STANDARD_PARAMETERS[name])
              for name in param_names}
    for value, limit, param in zip(values, limits, params.values()):
        param.set(*limit)
        param.value = value
    shared_params = {"p": params["p"]}
    del params["p"]

    components_and_params = {"Star": params,
                             "AsymmetricSDGreyBodyContinuum": shared_params}

    theta = mcmc.set_theta_from_params(components_and_params,
                                       shared_params)
    new_components_and_params, new_shared_parameters =\
        mcmc.set_params_from_theta(
            theta, components_and_params, shared_params)

    OPTIONS["model.components_and_params"] = components_and_params
    OPTIONS["model.shared_params"] = shared_params
    assert mcmc.lnprior(new_components_and_params,
                        new_shared_parameters) == expected

    OPTIONS["model.components_and_params"] = {}
    OPTIONS["model.shared_params"] = {}


def test_lnprob() -> None:
    ...


def test_run_mcmc() -> None:
    ...


def test_get_best_fit() -> None:
    ...
