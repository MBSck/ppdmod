import random
from typing import Dict, List
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest

from ppdmod import data
from ppdmod import mcmc
from ppdmod.custom_components import Star, AsymmetricSDGreyBodyContinuum
from ppdmod.model import Model
from ppdmod.parameter import STANDARD_PARAMETERS, Parameter
from ppdmod.options import OPTIONS


# TODO: Add here the calculation of the binning differences.

@pytest.fixture
def qval_file_dir() -> Path:
    """The qval-file directory."""
    return Path("data/qval")


@pytest.fixture
def wavelengths() -> u.um:
    """A wavelength grid."""
    return ([8.28835527e-06, 1.02322101e-05, 13.000458e-6]*u.m).to(u.um)


@pytest.fixture
def fits_files() -> Path:
    """A MATISSE (.fits)-file."""
    files = [
        "hd_142666_2022-04-23T03_05_25:2022-04-23T02_28_06_AQUARIUS_FINAL_TARGET_INT.fits",
        "hd_142666_2022-04-21T07_18_22:2022-04-21T06_47_05_AQUARIUS_FINAL_TARGET_INT.fits"]
    return [Path("data/fits") / file for file in files]


@pytest.fixture
def wavelength_solution(fits_files: Path) -> u.um:
    """The wavelength solution of a MATISSE (.fits)-file."""
    return (data.ReadoutFits(fits_files[0]).wavelength*u.m).to(u.um)


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
    return [[component, {key: Parameter(**STANDARD_PARAMETERS[key])
            for key in random.sample(list(STANDARD_PARAMETERS.keys()), 4)}]
            for component in components]


@pytest.fixture
def dim() -> int:
    """The dimension of the model."""
    return 2048


@pytest.fixture
def pixel_size() -> int:
    """The pixel size of the model."""
    return 0.1*u.mas


@pytest.fixture
def position_angle() -> u.deg:
    """The position angle of the model."""
    return 45*u.deg


@pytest.fixture
def axis_ratio() -> u.one:
    """The axis ratio of the model."""
    return 1.6


@pytest.fixture
def model(dim: int, pixel_size: u.mas,
          axis_ratio: u.one, position_angle: u.deg) -> Model:
    """Creates a model."""
    static_params = {"dim": dim, "dist": 145,
                     "eff_temp": 7800, "eff_radius": 1.8}
    star = Star(**static_params)
    inner_ring = AsymmetricSDGreyBodyContinuum(
        **static_params,
        rin=25, rout=30, a=0.3, phi=33,
        pixel_size=pixel_size.value, pa=position_angle,
        elong=axis_ratio, inner_sigma=2000, kappa_abs=1000,
        kappa_cont=3000, cont_weight=0.5, p=0.5)
    inner_ring.optically_thick = True
    outer_ring = AsymmetricSDGreyBodyContinuum(
        **static_params,
        rin=33, a=0.3, phi=33,
        pixel_size=pixel_size.value, pa=position_angle,
        elong=axis_ratio, inner_sigma=2000, kappa_abs=1000,
        kappa_cont=3000, cont_weight=0.5, p=0.5)
    outer_ring.optically_thick = True
    return Model([star, inner_ring, outer_ring])


def test_set_theta_from_params(
        components_and_params: Dict[str, Dict],
        shared_params: Dict[str, Parameter]) -> None:
    """Tests the set_theta_from_params function."""
    theta = mcmc.set_theta_from_params(components_and_params,
                                       shared_params)
    len_params = sum(len(params) for (_, params) in components_and_params)
    assert theta.size == len_params+len(shared_params)
    assert all(theta[-len(shared_params):] ==
               [parameter.value for parameter in shared_params.values()])


# TODO: Test order of appearance for parameters.
def test_set_params_from_theta(
        components_and_params: Dict[str, Dict],
        shared_params: Dict[str, Parameter]) -> None:
    """Tests the set_params_from_theta function."""
    OPTIONS["model.components_and_params"] = components_and_params
    OPTIONS["model.shared_params"] = shared_params
    theta = mcmc.set_theta_from_params(components_and_params,
                                       shared_params)
    new_components_and_params, new_shared_parameters =\
        mcmc.set_params_from_theta(theta)
    all_params = []
    for (_, params) in components_and_params:
        all_params.extend(list(map(lambda x: x.value, params.values())))
    new_params = []
    for (_, params) in new_components_and_params:
        new_params.extend(list(params.values()))
    assert all_params == new_params
    assert list(map(lambda x: x.value, shared_params.values())) ==\
        list(new_shared_parameters.values())

    OPTIONS["model.components_and_params"] = []
    OPTIONS["model.shared_params"] = []


# TODO: Tests somehow that all components end up where they need to go.
@pytest.mark.parametrize("nwalkers", [5, 10, 25, 35])
def test_init_randomly(nwalkers: int) -> None:
    """Tests the init_randomly function."""
    values = [1.5, 0.5, 0.3, 33, 0.2, 45, 1.6]
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

    components_and_params = [["Star", params],
                             ["AsymmetricSDGreyBodyContinuum", params]]

    OPTIONS["model.components_and_params"] = components_and_params
    OPTIONS["model.shared_params"] = shared_params

    theta = mcmc.init_randomly(nwalkers)
    assert theta.shape == (nwalkers, len(param_names)*2-1)

    OPTIONS["model.components_and_params"] = []
    OPTIONS["model.shared_params"] = []


# TODO: Finish test.
# @pytest.mark.parametrize(
#     "data, error, data_model, expected",
#     [(np.full((6,), fill_value=50), np.array([50, 60, 40, 47, 53]), 4.36)])
# def test_chi_sq(data: np.ndarray, error: np.ndarray,
#                 data_model: np.ndarray, expected: int) -> None:
#     """Tests the chi squre function."""
#     assert chi_sq(data, error, data_model) == expected


def test_calculate_observables(model: Model,
                               fits_files: List[Path],
                               pixel_size: u.mas,
                               wavelengths: u.um) -> None:
    """Tests the calculate_observables function."""
    data.set_fit_wavelengths(wavelengths)
    data.set_data(fits_files)

    readout = data.ReadoutFits(fits_files[0])
    total_flux, corr_flux, cphase = mcmc.calculate_observables(
        model.calculate_complex_visibility(wavelengths[0]),
        readout.ucoord, readout.vcoord,
        readout.u123coord, readout.v123coord,
        pixel_size, wavelengths[0])

    assert total_flux is not None
    assert corr_flux is not None
    assert cphase is not None

    assert total_flux.dtype == float
    assert corr_flux.dtype == float
    assert cphase.dtype == float

    assert total_flux.shape == ()
    assert corr_flux.shape == (6,)
    assert cphase.shape == (4,)

    data.set_fit_wavelengths()
    data.set_data()


def test_calculate_observables_chi_sq(
        model: Model, fits_files: List[Path],
        pixel_size: u.mas, wavelengths: u.um) -> None:
    """Tests the calculate_observables chi sq function."""
    data.set_fit_wavelengths(wavelengths)
    data.set_data(fits_files)

    readout = data.ReadoutFits(fits_files[0])
    total_flux_model, corr_flux_model, cphase_model =\
        mcmc.calculate_observables(
            model.calculate_complex_visibility(wavelengths[0]),
            readout.ucoord, readout.vcoord,
            readout.u123coord, readout.v123coord,
            pixel_size, wavelengths[0])

    wavelength_str = str(wavelengths[0].value)
    total_fluxes, total_fluxes_err =\
        OPTIONS["data.total_flux"], OPTIONS["data.total_flux_error"]
    corr_fluxes, corr_fluxes_err =\
        OPTIONS["data.correlated_flux"], OPTIONS["data.correlated_flux_error"]
    cphases, cphases_err =\
        OPTIONS["data.closure_phase"], OPTIONS["data.closure_phase_error"]

    chi_sq = 0
    for total_flux, total_flux_err, corr_flux,\
            corr_flux_err, cphase, cphase_err\
            in zip(total_fluxes, total_fluxes_err,
                   corr_fluxes, corr_fluxes_err, cphases, cphases_err):

        chi_sq += mcmc.calculate_observables_chi_sq(
            total_flux[wavelength_str], total_flux_err[wavelength_str],
            total_flux_model,
            corr_flux[wavelength_str], corr_flux_err[wavelength_str],
            corr_flux_model,
            cphase[wavelength_str], cphase_err[wavelength_str],
            cphase_model)

    assert chi_sq != 0
    assert isinstance(chi_sq, float)

    data.set_fit_wavelengths()
    data.set_data()


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

    components_and_params = [["Star", params],
                             ["AsymmetricSDGreyBodyContinuum", shared_params]]

    OPTIONS["model.components_and_params"] = components_and_params
    OPTIONS["model.shared_params"] = shared_params

    theta = mcmc.set_theta_from_params(components_and_params,
                                       shared_params)
    new_components_and_params, new_shared_parameters =\
        mcmc.set_params_from_theta(theta)

    assert mcmc.lnprior(new_components_and_params,
                        new_shared_parameters) == expected

    OPTIONS["model.components_and_params"] = []
    OPTIONS["model.shared_params"] = []


def test_lnprob(fits_files: List[Path], wavelengths: u.um) -> None:
    """Tests the lnprob function."""
    static_params = {"dim": 2048, "dist": 145, "pixel_size": 0.1,
                     "eff_temp": 7800, "eff_radius": 1.8,
                     "kappa_abs": 1000, "kappa_cont": 3000}
    param_names = ["rin", "p", "a", "phi",
                   "inner_sigma", "cont_weight", "pa", "elong"]
    values = [1.5, 0.5, 0.3, 33, 2000, 0.5, 45, 1.6]
    limits = [[0, 20], [0, 1], [0, 1], [0, 360],
              [100, 10000], [0, 1], [0, 360], [1, 50]]
    params = {name: Parameter(**STANDARD_PARAMETERS[name])
              for name in param_names}

    for value, limit, param in zip(values, limits, params.values()):
        param.set(*limit)
        param.value = value

    shared_params = {"p": params["p"], "pa": params["pa"],
                     "elong": params["elong"]}
    del params["p"]
    del params["pa"]
    del params["elong"]

    components_and_params = [["Star", params],
                             ["AsymmetricSDGreyBodyContinuum", params]]

    data.set_fit_wavelengths(wavelengths)
    data.set_data(fits_files)
    OPTIONS["model.constant_params"] = static_params
    OPTIONS["model.components_and_params"] = components_and_params
    OPTIONS["model.shared_params"] = shared_params

    theta = mcmc.set_theta_from_params(
        components_and_params, shared_params)

    chi_sq = mcmc.lnprob(theta)
    assert chi_sq != 0
    assert isinstance(chi_sq, float)

    OPTIONS["model.components_and_params"] = []
    OPTIONS["model.shared_params"] = []
    data.set_fit_wavelengths()
    data.set_data()


# TODO: Finish test.
def test_run_mcmc() -> None:
    """Tests the run_mcmc function."""
    ...


def test_get_best_fit() -> None:
    """Tests the get_best_fit function."""
    ...
