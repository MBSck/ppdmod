from typing import Dict
from pathlib import Path

import astropy.units as u
import pytest

from ppdmod import mcmc
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
def parameters() -> Dict[str, Parameter]:
    """Parameters."""
    rin = Parameter(name="rin", value=0.5, unit=u.mas,
                    description="Inner radius of the disk")
    inner_sigma = Parameter(name="inner_sigma", value=100,
                            unit=u.g/u.cm**2, free=False,
                            description="Inner surface density")
    a = Parameter(name="a", value=0.5, unit=u.one,
                  description="Azimuthal modulation amplitude")
    phi = Parameter(name="phi", value=33, unit=u.deg,
                    description="Azimuthal modulation angle")
    cont_weight = Parameter(name="cont_weight", value=0.6,
                            unit=u.one, free=True,
                            description="Dust mass continuum absorption coefficient's weight")
    pa = Parameter(**STANDARD_PARAMETERS["pa"])
    elong = Parameter(**STANDARD_PARAMETERS["elong"])

    rin.set(min=0, max=20)
    a.set(min=0., max=1.)
    phi.set(min=0, max=360)
    cont_weight.set(min=0, max=1)
    pa.set(min=0, max=360)
    elong.set(min=1, max=50)
    return {"rin": rin, "a": a, "phi": phi,
            "cont_weight": cont_weight, "pa": pa, "elong": elong}


@pytest.fixture
def shared_parameters() -> Dict[str, Parameter]:
    """Shared parameters."""
    p = Parameter(name="p", value=0.5, unit=u.one,
                  description="Power-law exponent for the surface density profile")
    p.set(min=0., max=1.)
    return {"p": p}


# TODO: Add more tests for set_theta_from_params and set_params_from_theta.
def test_set_theta_from_params(
        parameters: Dict[str, Parameter],
        shared_parameters: Dict[str, Parameter]) -> None:
    """Tests the set_theta_from_params function."""
    components_and_params = {"Star": parameters,
                             "AsymmetricSDGreyBodyContinuum": parameters}
    theta = mcmc.set_theta_from_params(components_and_params, shared_parameters)
    len_params = sum(len(params) for params in components_and_params.values())
    assert theta.size == len_params+len(shared_parameters)
    assert all(theta[-len(shared_parameters):] ==
               [parameter.value for parameter in shared_parameters.values()])


# TODO: Test for multiple components and also varied parameters.
def test_set_params_from_theta(
        parameters: Dict[str, Parameter],
        shared_parameters: Dict[str, Parameter]) -> None:
    """Tests the set_params_from_theta function."""
    components_and_params = {"Star": parameters,
                             "AsymmetricSDGreyBodyContinuum": parameters}
    theta = mcmc.set_theta_from_params(components_and_params, shared_parameters)
    new_components_and_params, new_shared_parameters =\
        mcmc.set_params_from_theta(
            theta, components_and_params, shared_parameters)
    all_params = []
    for component, params in components_and_params.items():
        all_params.extend(list(map(lambda x: x.value, params.values())))
    new_params = []
    for component, params in new_components_and_params.items():
        new_params.extend(list(params.values()))
    assert all_params == new_params
    assert list(map(lambda x: x.value, shared_parameters.values())) ==\
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


def test_lnprob() -> None:
    ...


def test_run_mcmc() -> None:
    ...
