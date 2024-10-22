import random
from pathlib import Path
from typing import Dict, List, Tuple

import astropy.units as u
import numpy as np
import pytest

from ppdmod import fitting
from ppdmod.basic_components import assemble_components
from ppdmod.data import set_data
from ppdmod.options import OPTIONS, STANDARD_PARAMETERS
from ppdmod.parameter import Parameter


DATA_DIR = Path(__file__).parent.parent / "data"

@pytest.fixture
def fits_files() -> list[Path]:
    """A MATISSE (.fits)-file."""
    return list((DATA_DIR / "matisse").glob("*.fits"))


@pytest.fixture
def mock_components_and_params() -> Dict[str, Dict]:
    """Mock parameters connected to their components."""
    all_components = [
        "Star",
        "GreyBody",
        "AsymmetricGreyBody",
        "TempGradient",
        "AsymmetricTemperatureGradient",
    ]
    components = random.sample(all_components, 3)
    return [
        [
            component,
            {
                key: Parameter(**getattr(STANDARD_PARAMETERS, key))
                for key in random.sample(list(vars(STANDARD_PARAMETERS).keys()), 4)
            },
        ]
        for component in components
    ]


@pytest.fixture
def mock_shared_params() -> Dict[str, Parameter]:
    """Mock shared parameters."""
    return {
        key: Parameter(**getattr(STANDARD_PARAMETERS, key))
        for key in random.sample(list(vars(STANDARD_PARAMETERS).keys()), 4)
    }


@pytest.fixture
def constant_params() -> Dict:
    return {
        "dim": 2048,
        "dist": 148.3,
        "eff_temp": 7500,
        "eff_radius": 1.75,
        "kappa_abs": 1000,
        "kappa_cont": 3000,
    }


@pytest.fixture
def components_and_params() -> List[Dict[str, Parameter]]:
    """Parameters connected to their components."""
    rin = Parameter(**STANDARD_PARAMETERS.rin)
    rout = Parameter(**STANDARD_PARAMETERS.rout)
    p = Parameter(**STANDARD_PARAMETERS.p)
    c = Parameter(**STANDARD_PARAMETERS.c)
    s = Parameter(**STANDARD_PARAMETERS.s)
    sigma0 = Parameter(**STANDARD_PARAMETERS.sigma0)

    rin.value = 1.0
    rout.value = 7.0
    c.value = 0.5
    s.value = 0.5
    p.value = 0.5
    sigma0.value = 1e-3

    rin.set(min=0.5, max=5)
    rout.set(min=0.5, max=10)
    c.set(min=-1.0, max=1.0)
    s.set(min=-1, max=1)
    p.set(min=0.0, max=1.0)
    sigma0.set(min=0, max=1e-2)

    rout.free = True

    inner_ring = {"rin": rin, "rout": rout, "sigma0": sigma0, "p": p, "c": c, "s": s}
    return [["Star", {}], ["AsymmetricGreyBody", inner_ring]]


@pytest.fixture
def shared_params() -> Dict:
    """Shared parameters."""
    pa = Parameter(**STANDARD_PARAMETERS.pa)
    inc = Parameter(**STANDARD_PARAMETERS.inc)
    cont_weight = Parameter(**STANDARD_PARAMETERS.cont_weight)

    pa.value = 145
    inc.value = 0.5
    cont_weight.value = 0.4

    pa.set(min=0, max=360)
    inc.set(min=0, max=1)
    cont_weight.set(min=0.3, max=0.8)
    return {"pa": pa, "inc": inc, "cont_weight": cont_weight}


def test_set_theta_from_params(
    components_and_params: Dict[str, Dict], shared_params: Dict[str, Parameter]
) -> None:
    """Tests the set_theta_from_params function."""
    theta = fitting.set_theta_from_params(components_and_params, shared_params)
    len_params = sum(len(params) for (_, params) in components_and_params)
    assert theta.size == len_params + len(shared_params)
    assert all(
        theta[-len(shared_params) :]
        == [parameter.value for parameter in shared_params.values()]
    )


# TODO: Test if order is kept for parameters
def test_set_params_from_theta(
    mock_components_and_params: Dict[str, Dict],
    mock_shared_params: Dict[str, Parameter],
) -> None:
    """Tests the set_params_from_theta function."""
    OPTIONS.model.components_and_params = mock_components_and_params
    OPTIONS.model.shared_params = mock_shared_params
    theta = fitting.set_theta_from_params(
        mock_components_and_params, mock_shared_params
    )
    new_components_and_params, new_shared_parameters = fitting.set_params_from_theta(
        theta
    )
    all_params = []
    for _, params in mock_components_and_params:
        all_params.extend(list(map(lambda x: x.value, params.values())))
    new_params = []
    for _, params in new_components_and_params:
        new_params.extend(list(params.values()))
    assert all_params == new_params
    assert list(map(lambda x: x.value, mock_shared_params.values())) == list(
        new_shared_parameters.values()
    )

    OPTIONS.model.components_and_params = {}
    OPTIONS.model.shared_params = {}


# TODO: Tests somehow that all components end up where they need to go.
@pytest.mark.parametrize("nwalkers", [5, 10, 25, 35])
def test_init_randomly(nwalkers: int) -> None:
    """Tests the init_randomly function."""
    values = [1.5, 0.5, 0.3, 0.7, 0.2, 45, 1.6]
    param_names = ["rin", "p", "c", "s", "cont_weight", "pa", "inc"]
    limits = [[0, 20], [0, 1], [-1, 1], [-1, 1], [0, 1], [0, 360], [1, 50]]
    params = {
        name: Parameter(**getattr(STANDARD_PARAMETERS, name)) for name in param_names
    }
    for value, limit, param in zip(values, limits, params.values()):
        param.set(*limit)
        param.value = value

    shared_params = {"p": params["p"]}
    del params["p"]

    components_and_params = [["Star", params], ["AsymmetricGreyBody", params]]

    OPTIONS.model.components_and_params = components_and_params
    OPTIONS.model.shared_params = shared_params

    theta = fitting.init_randomly(nwalkers)
    assert theta.shape == (nwalkers, len(param_names) * 2 - 1)

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


@pytest.mark.parametrize(
    "wavelength", [[3.5] * u.um, [8] * u.um, [3.5, 8] * u.um, [3.5, 8, 10] * u.um]
)
def test_calculate_observables(
    components_and_params: List[Tuple[str, Dict]],
    shared_params: Dict[str, Parameter],
    constant_params: Dict[str, Parameter],
    fits_files: List[Path],
    wavelength: u.um,
) -> None:
    """Tests the calculate_observables function."""
    data = set_data(fits_files, wavelengths=wavelength, fit_data=["flux", "vis", "t3"])
    nwl = wavelength.size

    OPTIONS.model.components_and_params = components_and_params
    OPTIONS.model.shared_params = shared_params
    OPTIONS.model.constant_params = constant_params
    OPTIONS.model.modulation = 1

    flux, vis, t3 = fitting.compute_observables(
        assemble_components(components_and_params, shared_params), wavelength
    )

    assert flux is not None and vis is not None and t3 is not None
    assert flux.dtype == vis.dtype == t3.dtype == data.dtype.real
    assert flux.shape == (nwl, len(data.readouts))
    assert vis.shape == (nwl, data.vis.vcoord.size - 1)
    assert t3.shape == (nwl, data.t3.u123coord.shape[1] - 1)

    set_data()

    OPTIONS.model.components_and_params = []
    OPTIONS.model.shared_params = {}
    OPTIONS.model.constant_params = {}
    OPTIONS.model.modulation = 0


@pytest.mark.parametrize(
    "wavelength", [[3.5] * u.um, [8] * u.um, [3.5, 8] * u.um, [3.5, 8, 10] * u.um]
)
def test_calculate_chi_sq(
    components_and_params: List[Tuple[str, Dict]],
    shared_params: Dict[str, Parameter],
    constant_params: Dict[str, Parameter],
    fits_files: List[Path],
    wavelength: u.um,
) -> None:
    """Tests the calculate_observables chi sq function."""
    set_data(fits_files, wavelengths=wavelength)

    OPTIONS.model.components_and_params = components_and_params
    OPTIONS.model.shared_params = shared_params
    OPTIONS.model.constant_params = constant_params
    OPTIONS.model.modulation = 1

    components = assemble_components(components_and_params, shared_params)
    chi_sq = fitting.compute_observable_chi_sq(*fitting.compute_observables(components))

    assert chi_sq != 0
    assert isinstance(chi_sq, float)

    set_data()

    OPTIONS.model.components_and_params = []
    OPTIONS.model.shared_params = {}
    OPTIONS.model.constant_params = {}
    OPTIONS.model.modulation = 0


@pytest.mark.parametrize(
    "values, expected",
    [
        ([1.5, 0.5, 0.3, 0.7, 0.2, 45, 1.6], 0.0),
        ([1.5, 0.5, 0.3, 0.7, 0.2, 45, 0.6], -np.inf),
    ],
)
def test_lnprior(values: List[float], expected: float) -> None:
    """Tests the lnprior function."""
    param_names = ["rin", "p", "c", "s", "cont_weight", "pa", "inc"]
    limits = [[0, 20], [0, 1], [-1, 1], [-1, 1], [0, 1], [0, 360], [1, 50]]
    params = {
        name: Parameter(**getattr(STANDARD_PARAMETERS, name)) for name in param_names
    }
    for value, limit, param in zip(values, limits, params.values()):
        param.set(*limit)
        param.value = value
    shared_params = {"p": params["p"]}
    del params["p"]

    components_and_params = [["Star", params], ["AsymmetricGreyBody", shared_params]]

    OPTIONS.model.components_and_params = components_and_params
    OPTIONS.model.shared_params = shared_params
    OPTIONS.model.modulation = 1

    theta = fitting.set_theta_from_params(components_and_params, shared_params)
    new_components_and_params, new_shared_parameters = fitting.set_params_from_theta(
        theta
    )

    assert fitting.lnprior(new_components_and_params, new_shared_parameters) == expected

    OPTIONS.model.components_and_params = []
    OPTIONS.model.shared_params = {}
    OPTIONS.model.constant_params = {}
    OPTIONS.model.modulation = 0


@pytest.mark.parametrize(
    "wavelength", [[3.5] * u.um, [8] * u.um, [3.5, 8] * u.um, [3.5, 8, 10] * u.um]
)
def test_lnprob(fits_files: List[Path], wavelength: u.um) -> None:
    """Tests the lnprob function."""
    static_params = {
        "dim": 2048,
        "dist": 145,
        "pixel_size": 0.1,
        "eff_temp": 7800,
        "eff_radius": 1.8,
        "kappa_abs": 1000,
        "kappa_cont": 3000,
    }
    param_names = ["rin", "p", "c", "s", "sigma0", "cont_weight", "pa", "inc"]
    values = [1.5, 0.5, 0.3, 0.7, 2000, 0.5, 45, 1.6]
    limits = [
        [0, 20],
        [0, 1],
        [-1, 1],
        [-1, 1],
        [100, 10000],
        [0, 1],
        [0, 360],
        [1, 50],
    ]
    params = {
        name: Parameter(**getattr(STANDARD_PARAMETERS, name)) for name in param_names
    }

    for value, limit, param in zip(values, limits, params.values()):
        param.set(*limit)
        param.value = value

    shared_params = {"p": params["p"], "pa": params["pa"], "inc": params["inc"]}
    del params["p"]
    del params["pa"]
    del params["inc"]

    components_and_params = [["Star", params], ["AsymmetricGreyBody", params]]

    set_data(fits_files, wavelengths=wavelength)
    OPTIONS.model.constant_params = static_params
    OPTIONS.model.components_and_params = components_and_params
    OPTIONS.model.shared_params = shared_params
    OPTIONS.model.modulation = 1

    theta = fitting.set_theta_from_params(components_and_params, shared_params)

    chi_sq = fitting.lnprob(theta)
    assert isinstance(chi_sq, float)
    assert chi_sq != 0

    set_data()

    OPTIONS.model.components_and_params = []
    OPTIONS.model.shared_params = {}
    OPTIONS.model.constant_params = {}
    OPTIONS.model.modulation = 0
