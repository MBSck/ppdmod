from pathlib import Path
from typing import Dict, List, Tuple

import astropy.units as u
import numpy as np
import pytest

from ppdmod.basic_components import AsymGreyBody, Star
from ppdmod.component import Component
from ppdmod.data import set_data
from ppdmod.fitting import (
    compute_interferometric_chi_sq,
    compute_observables,
    get_labels,
    get_theta,
    lnprob,
    set_components_from_theta,
)
from ppdmod.options import OPTIONS
from ppdmod.parameter import Parameter

DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture
def fits_files() -> list[Path]:
    """A MATISSE (.fits)-file."""
    return list((DATA_DIR / "matisse").glob("*.fits"))


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
def shared_params() -> Dict:
    """Shared parameters."""
    pa = Parameter(value=145, min=0, max=360, free=True, shared=True, base="pa")
    inc = Parameter(value=0.5, min=0, max=1, free=True, shared=True, base="inc")
    cont_weight = Parameter(
        value=40, min=0.3, max=0.8, free=True, shared=True, base="weight_cont"
    )
    return {"pa": pa, "inc": inc, "cont_weight": cont_weight}


@pytest.fixture
def components(shared_params: Dict[str, Parameter]) -> List[Component]:
    """The model components."""
    rin = Parameter(value=1, min=0.5, max=5, base="rin")
    rout = Parameter(value=7, min=0.5, max=10, free=True, base="rout")
    p = Parameter(value=0.3, min=-1, max=1, base="p")
    c = Parameter(value=0.9, min=0, max=1, base="c")
    s = Parameter(value=0.6, base="s")
    sigma0 = Parameter(value=1e-3, min=0, max=1e-2, base="sigma0")

    inner_ring = AsymGreyBody(
        rin=rin, rout=rout, p=p, sigma0=sigma0, c1=c, s1=s, **shared_params
    )
    return [Star(), inner_ring]


def test_get_priors(): ...


def test_get_theta(
    components: List[Component], shared_params: Dict[str, Parameter]
) -> None:
    """Tests the get_theta function."""
    theta = get_theta(components)
    breakpoint()
    # len_params = sum(len(params) for (_, params) in components)
    assert theta.size == len_params + len(shared_params)
    assert all(
        theta[-len(shared_params) :]
        == [parameter.value for parameter in shared_params.values()]
    )


def test_set_params_from_theta(theta: np.ndarray) -> None:
    """Tests the set_params_from_theta function."""
    OPTIONS.model.components = mock_components_and_params
    theta = set_theta_from_params(mock_components_and_params, mock_shared_params)
    new_components_and_params, new_shared_parameters = set_components_from_theta(theta)
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

    OPTIONS.model.components = {}


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
    components: List[Component],
    fits_files: List[Path],
    wavelength: u.um,
) -> None:
    """Tests the calculate_observables function."""
    data = set_data(fits_files, wavelengths=wavelength, fit_data=["flux", "vis", "t3"])
    nwl = wavelength.size

    OPTIONS.model.components_and_params = components
    OPTIONS.model.modulation = 1

    flux, vis, t3 = compute_observables()

    assert flux is not None and vis is not None and t3 is not None
    assert flux.dtype == vis.dtype == t3.dtype == data.dtype.real
    assert flux.shape == (nwl, len(data.readouts))
    assert vis.shape == (nwl, data.vis.vcoord.size - 1)
    assert t3.shape == (nwl, data.t3.u123coord.shape[1] - 1)

    set_data()

    OPTIONS.model.components_and_params = []
    OPTIONS.model.modulation = 0


@pytest.mark.parametrize(
    "wavelength", [[3.5] * u.um, [8] * u.um, [3.5, 8] * u.um, [3.5, 8, 10] * u.um]
)
def test_calculate_chi_sq(
    components_and_params: List[Tuple[str, Dict]],
    fits_files: List[Path],
    wavelength: u.um,
) -> None:
    """Tests the calculate_observables chi sq function."""
    set_data(fits_files, wavelengths=wavelength)

    OPTIONS.model.components_and_params = components_and_params
    OPTIONS.model.constant_params = constant_params
    OPTIONS.model.modulation = 1

    components = set_components_from_theta(components)
    chi_sq = compute_interferometric_chi_sq(*compute_observables(components))

    assert chi_sq != 0
    assert isinstance(chi_sq, float)

    set_data()

    OPTIONS.model.components_and_params = []
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
    params = {name: Parameter(base=name) for name in param_names}
    for index, param in enumerate(params.values()):
        param.min, param.max = limits[index]
        param.value = values[index]
        param.free = True
        if param.shortname in ["p", "pa", "inc"]:
            param.shared = True

    shared_params = {"p": params["p"], "pa": params["pa"], "inc": params["inc"]}
    del params["p"]
    del params["pa"]
    del params["inc"]

    OPTIONS.model.components = components = [Star(), AsymGreyBody(**params)]
    OPTIONS.model.modulation = 1

    set_data(fits_files, wavelengths=wavelength)
    theta = get_theta(components)

    chi_sq = lnprob(theta)
    assert isinstance(chi_sq, float)
    assert chi_sq != 0

    set_data()
    OPTIONS.model.components_and_params = []
    OPTIONS.model.modulation = 0
