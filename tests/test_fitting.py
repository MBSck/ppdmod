import random
from typing import Tuple, Dict, List
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest

from ppdmod import data
from ppdmod import fitting
from ppdmod.custom_components import assemble_components
from ppdmod.parameter import Parameter
from ppdmod.options import OPTIONS, STANDARD_PARAMETERS


@pytest.fixture
def qval_file_dir() -> Path:
    """The qval-file directory."""
    return Path("data/qval")


@pytest.fixture
def wavelengths() -> u.um:
    """A wavelength grid."""
    return [3.5, 10, 12.5]*u.um


@pytest.fixture
def fits_files() -> Path:
    """A MATISSE (.fits)-file."""
    day_two = list(Path("data/fits").glob("*2022-04-23*.fits"))
    day_one = list(Path("data/fits").glob("*2022-04-21*.fits"))
    return day_two + day_one


@pytest.fixture
def wavelength_solution(fits_files: Path) -> u.um:
    """The wavelength solution of a MATISSE (.fits)-file."""
    return data.ReadoutFits(fits_files[0]).wavelength*u.um


@pytest.fixture
def mock_components_and_params() -> Dict[str, Dict]:
    """Mock parameters connected to their components."""
    all_components = ["Star", "GreyBody",
                      "AsymmetricGreyBody", "TempGradient",
                      "AsymmetricTemperatureGradient"]
    components = random.sample(all_components, 3)
    return [[component, {key: Parameter(**STANDARD_PARAMETERS[key])
            for key in random.sample(list(STANDARD_PARAMETERS.keys()), 4)}]
            for component in components]


@pytest.fixture
def mock_shared_params() -> Dict[str, Parameter]:
    """Mock shared parameters."""
    return {key: Parameter(**STANDARD_PARAMETERS[key])
            for key in random.sample(list(STANDARD_PARAMETERS.keys()), 4)}


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
    return 1.6*u.one


@pytest.fixture
def constant_params(dim: int) -> Dict:
    return {"dim": dim, "dist": 148.3,
            "eff_temp": 7500, "eff_radius": 1.75,
            "kappa_abs": 1000, "kappa_cont": 3000}


@pytest.fixture
def components_and_params() -> List[Dict[str, Parameter]]:
    """Parameters connected to their components."""
    rin = Parameter(**STANDARD_PARAMETERS["rin"])
    rout = Parameter(**STANDARD_PARAMETERS["rout"])
    p = Parameter(**STANDARD_PARAMETERS["p"])
    a = Parameter(**STANDARD_PARAMETERS["a"])
    phi = Parameter(**STANDARD_PARAMETERS["phi"])
    inner_sigma = Parameter(**STANDARD_PARAMETERS["inner_sigma"])

    rin.value = 1.
    rout.value = 7.
    a.value = 0.5
    phi.value = 133
    p.value = 0.5
    inner_sigma.value = 1e-3

    rin.set(min=0.5, max=5)
    rout.set(min=0.5, max=10)
    a.set(min=0., max=1.)
    phi.set(min=0, max=360)
    p.set(min=0., max=1.)
    inner_sigma.set(min=0, max=1e-2)

    rout.free = True

    inner_ring = {"rin": rin, "rout": rout,
                  "inner_sigma": inner_sigma, "p": p,
                  "a": a, "phi": phi}
    return [["Star", {}], ["AsymmetricGreyBody", inner_ring]]


@pytest.fixture
def shared_params() -> Dict:
    """Shared parameters."""
    pa = Parameter(**STANDARD_PARAMETERS["pa"])
    elong = Parameter(**STANDARD_PARAMETERS["elong"])
    cont_weight = Parameter(**STANDARD_PARAMETERS["cont_weight"])

    pa.value = 145
    elong.value = 0.5
    cont_weight.value = 0.4

    pa.set(min=0, max=360)
    elong.set(min=0, max=1)
    cont_weight.set(min=0.3, max=0.8)
    return {"pa": pa, "elong": elong, "cont_weight": cont_weight}


def test_set_theta_from_params(
        components_and_params: Dict[str, Dict],
        shared_params: Dict[str, Parameter]) -> None:
    """Tests the set_theta_from_params function."""
    theta = fitting.set_theta_from_params(components_and_params,
                                       shared_params)
    len_params = sum(len(params) for (_, params) in components_and_params)
    assert theta.size == len_params+len(shared_params)
    assert all(theta[-len(shared_params):] ==
               [parameter.value for parameter in shared_params.values()])


# TODO: Test if order is kept for parameters
def test_set_params_from_theta(
        mock_components_and_params: Dict[str, Dict],
        mock_shared_params: Dict[str, Parameter]) -> None:
    """Tests the set_params_from_theta function."""
    OPTIONS.model.components_and_params = mock_components_and_params
    OPTIONS.model.shared_params = mock_shared_params
    theta = fitting.set_theta_from_params(mock_components_and_params,
                                       mock_shared_params)
    new_components_and_params, new_shared_parameters =\
        fitting.set_params_from_theta(theta)
    all_params = []
    for (_, params) in mock_components_and_params:
        all_params.extend(list(map(lambda x: x.value, params.values())))
    new_params = []
    for (_, params) in new_components_and_params:
        new_params.extend(list(params.values()))
    assert all_params == new_params
    assert list(map(lambda x: x.value, mock_shared_params.values())) ==\
        list(new_shared_parameters.values())

    OPTIONS.model.components_and_params = {}
    OPTIONS.model.shared_params = {}


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
                             ["AsymmetricGreyBody", params]]

    OPTIONS.model.components_and_params = components_and_params
    OPTIONS.model.shared_params = shared_params

    theta = fitting.init_randomly(nwalkers)
    assert theta.shape == (nwalkers, len(param_names)*2-1)

    OPTIONS.model.components_and_params = []
    OPTIONS.model.shared_params = []


# TODO: Finish test.
# TODO: Test exponential chi_sq.
# @pytest.mark.parametrize(
#     "data, error, data_model, expected",
#     [(np.full((6,), fill_value=50), np.array([50, 60, 40, 47, 53]), 4.36)])
# def test_chi_sq(data: np.ndarray, error: np.ndarray,
#                 data_model: np.ndarray, expected: int) -> None:
#     """Tests the chi squre function."""
#     assert chi_sq(data, error, data_model) == expected

# TODO: Test for different modulation orders
# TODO: Test if modulation is properly calculated in case of symmetrical and asymmetrical
# components.
# TODO: Test with bigger kappas also
@pytest.mark.parametrize(
        "wavelength", [[3.5]*u.um, [8]*u.um,
                       [3.5, 8]*u.um, [3.5, 8, 10]*u.um])
def test_calculate_observables(components_and_params: List[Tuple[str, Dict]],
                               shared_params: Dict[str, Parameter],
                               constant_params: Dict[str, Parameter],
                               fits_files: List[Path], wavelength: u.um) -> None:
    """Tests the calculate_observables function."""
    data.set_fit_wavelengths(wavelength)
    data.set_data(fits_files)
    nwl, nfile = wavelength.size, len(fits_files)

    OPTIONS.model.components_and_params = components_and_params
    OPTIONS.model.shared_params = shared_params
    OPTIONS.model.constant_params = constant_params
    OPTIONS.model.modulation = 1

    flux_model, vis_model, t3_model = fitting.calculate_observables(
        assemble_components(components_and_params, shared_params))

    assert flux_model is not None
    assert vis_model is not None
    assert t3_model is not None

    assert flux_model.dtype == OPTIONS.data.dtype.real
    assert vis_model.dtype == OPTIONS.data.dtype.real
    assert t3_model.dtype == OPTIONS.data.dtype.real

    assert flux_model.shape == (nwl, nfile)
    assert vis_model.shape == (nwl, nfile*6)
    assert t3_model.shape == (nwl, nfile*4)

    data.set_fit_wavelengths()
    data.set_data()

    OPTIONS.model.components_and_params = []
    OPTIONS.model.shared_params = {}
    OPTIONS.model.constant_params = {}
    OPTIONS.model.modulation = 0


@pytest.mark.parametrize(
        "wavelength", [[3.5]*u.um, [8]*u.um,
                       [3.5, 8]*u.um, [3.5, 8, 10]*u.um])
def test_calculate_chi_sq(components_and_params: List[Tuple[str, Dict]],
                          shared_params: Dict[str, Parameter],
                          constant_params: Dict[str, Parameter],
                          fits_files: List[Path], wavelength: u.um) -> None:
    """Tests the calculate_observables chi sq function."""
    data.set_fit_wavelengths(wavelength)
    data.set_data(fits_files)

    OPTIONS.model.components_and_params = components_and_params
    OPTIONS.model.shared_params = shared_params
    OPTIONS.model.constant_params = constant_params
    OPTIONS.model.modulation = 1

    components = assemble_components(components_and_params, shared_params)
    chi_sq = fitting.calculate_observable_chi_sq(
            *fitting.calculate_observables(components))

    assert chi_sq != 0
    assert isinstance(chi_sq, float)

    data.set_fit_wavelengths()
    data.set_data()

    OPTIONS.model.components_and_params = []
    OPTIONS.model.shared_params = {}
    OPTIONS.model.constant_params = {}
    OPTIONS.model.modulation = 0


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
                             ["AsymmetricGreyBody", shared_params]]

    OPTIONS.model.components_and_params = components_and_params
    OPTIONS.model.shared_params = shared_params
    OPTIONS.model.modulation = 1

    theta = fitting.set_theta_from_params(components_and_params,
                                       shared_params)
    new_components_and_params, new_shared_parameters =\
            fitting.set_params_from_theta(theta)

    assert fitting.lnprior(new_components_and_params,
                           new_shared_parameters) == expected

    OPTIONS.model.components_and_params = []
    OPTIONS.model.shared_params = {}
    OPTIONS.model.constant_params = {}
    OPTIONS.model.modulation = 0


@pytest.mark.parametrize(
        "wavelength", [[3.5]*u.um, [8]*u.um,
                       [3.5, 8]*u.um, [3.5, 8, 10]*u.um])
def test_lnprob(fits_files: List[Path], wavelength: u.um) -> None:
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
                             ["AsymmetricGreyBody", params]]

    data.set_fit_wavelengths(wavelength)
    data.set_data(fits_files)
    OPTIONS.model.constant_params = static_params
    OPTIONS.model.components_and_params = components_and_params
    OPTIONS.model.shared_params = shared_params
    OPTIONS.model.modulation = 1

    theta = fitting.set_theta_from_params(
        components_and_params, shared_params)

    chi_sq = fitting.lnprob(theta)
    assert isinstance(chi_sq, float)
    assert chi_sq != 0

    data.set_fit_wavelengths()
    data.set_data()

    OPTIONS.model.components_and_params = []
    OPTIONS.model.shared_params = {}
    OPTIONS.model.constant_params = {}
    OPTIONS.model.modulation = 0


# # TODO: Finish test.
# def test_run_fit() -> None:
#     """Tests the run_fitting function."""
#     ...


# def test_get_best_fit() -> None:
#     """Tests the get_best_fit function."""
#     ...
