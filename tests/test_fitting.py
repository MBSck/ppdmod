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
    get_priors,
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
def shared_params() -> Dict[str, Parameter]:
    """Shared parameters."""
    pa = Parameter(value=145, min=0, max=360, free=True, shared=True, base="pa")
    cinc = Parameter(value=0.5, min=0, max=1, free=True, shared=True, base="cinc")
    weight_cont = Parameter(
        value=40, min=30, max=80, free=True, shared=True, base="weight_cont"
    )
    return {"pa": pa, "cinc": cinc, "weight_cont": weight_cont}


@pytest.fixture
def free_params() -> Dict[str, Parameter]:
    rin = Parameter(value=1, min=0.5, max=5, base="rin")
    rout = Parameter(value=7, min=0.5, max=10, free=True, base="rout")
    p = Parameter(value=0.3, min=-1, max=1, base="p")
    c = Parameter(value=0.9, min=0, max=1, base="c")
    s = Parameter(value=0.6, base="s")
    sigma0 = Parameter(value=1e-3, min=0, max=1e-2, base="sigma0")
    return {"rin": rin, "rout": rout, "p": p, "c1": c, "s1": s, "sigma0": sigma0}


@pytest.fixture
def components(
    free_params: Dict[str, Parameter], shared_params: Dict[str, Parameter]
) -> List[Component]:
    """The model components."""
    return [Star(), AsymGreyBody(**free_params, **shared_params)]


# TODO: Test all of the below with multiple components
def test_get_labels(
    components: List[Component],
    free_params: Dict[str, Parameter],
    shared_params: Dict[str, Parameter],
) -> None:
    nfree, nshared = len(free_params), len(shared_params)
    labels = get_labels(components)

    assert len(labels) == (nfree + nshared)

    zone_counter = 1
    for component in components:
        if component.get_params(shared=True):
            component_shared_labels = [
                label.split("-")[0] for label in labels[-nshared:]
            ]
            assert component_shared_labels == sorted(shared_params.keys())

            component_labels = [
                label.split("-")[0]
                for label in labels[:-nshared]
                if zone_counter == int(label.split("-")[-1])
            ]
            zone_counter += 1

            assert component_labels == sorted(free_params.keys())


# TODO: Finish this test
def test_get_priors(
    components: List[Component],
    free_params: Dict[str, Parameter],
    shared_params: Dict[str, Parameter],
) -> None:
    nfree, nshared = len(free_params), len(shared_params)
    priors = get_priors(components)

    assert priors.shape[0] == (nfree + nshared)

    for component in components:
        if component.get_params(shared=True):
            for index, (key, value) in enumerate(sorted(shared_params.items())):
                assert getattr(component, key) == value
                assert np.array_equal(
                    np.array([value.min, value.max]), priors[-nshared:][index]
                )

            # ncomponent_free = len(component.get_params(free=True))
            # assert component_labels == sorted(free_params.keys())

    # for component in
    #     assert
    #     breakpoint()


# TODO: Finish this test
def test_get_theta(components: List[Component]) -> None:
    """Tests the get_theta function."""
    # theta = get_theta(components)
    # len_params = sum(len(params) for (_, params) in components)
    # assert theta.size == len_params + len(shared_params)
    # assert all(
    #     theta[-len(shared_params) :]
    #     == [parameter.value for parameter in shared_params.values()]
    # )


def test_set_params_from_theta(
    components: List[Component],
    free_params: Dict[str, Parameter],
    shared_params: Dict[str, Parameter],
) -> None:
    """Tests the set_params_from_theta function."""
    nfree, nshared = len(free_params), len(shared_params)

    OPTIONS.model.components = components
    theta = np.array([0, 10, 2.33, 3, 4.245, 5235, 6.7, 7.3, 8.2])
    components = set_components_from_theta(theta)

    OPTIONS.model.components = {}


# # TODO: Finish test.
# # TODO: Test exponential chi_sq.
# # @pytest.mark.parametrize(
# #     "data, error, data_model, expected",
# #     [(np.full((6,), fill_value=50), np.array([50, 60, 40, 47, 53]), 4.36)])
# # def test_chi_sq(data: np.ndarray, error: np.ndarray,
# #                 data_model: np.ndarray, expected: int) -> None:
# #     """Tests the chi squre function."""
# #     assert chi_sq(data, error, data_model) == expected
#
#
# @pytest.mark.parametrize(
#     "wavelength", [[3.5] * u.um, [8] * u.um, [3.5, 8] * u.um, [3.5, 8, 10] * u.um]
# )
# def test_calculate_observables(
#     components: List[Component],
#     fits_files: List[Path],
#     wavelength: u.um,
# ) -> None:
#     """Tests the calculate_observables function."""
#     data = set_data(fits_files, wavelengths=wavelength, fit_data=["flux", "vis", "t3"])
#     nwl = wavelength.size
#
#     OPTIONS.model.components = components
#     OPTIONS.model.modulation = 1
#
#     flux, vis, t3 = compute_observables()
#
#     assert flux is not None and vis is not None and t3 is not None
#     assert flux.dtype == vis.dtype == t3.dtype == data.dtype.real
#     assert flux.shape == (nwl, len(data.readouts))
#     assert vis.shape == (nwl, data.vis.vcoord.size - 1)
#     assert t3.shape == (nwl, data.t3.u123coord.shape[1] - 1)
#
#     set_data()
#     OPTIONS.model.components = []
#     OPTIONS.model.modulation = 0
#
#
# @pytest.mark.parametrize(
#     "wavelength", [[3.5] * u.um, [8] * u.um, [3.5, 8] * u.um, [3.5, 8, 10] * u.um]
# )
# def test_calculate_chi_sq(
#     components: List[Tuple[str, Dict]],
#     fits_files: List[Path],
#     wavelength: u.um,
# ) -> None:
#     """Tests the calculate_observables chi sq function."""
#     set_data(fits_files, wavelengths=wavelength)
#
#     OPTIONS.model.components = components
#     OPTIONS.model.modulation = 1
#
#     components = set_components_from_theta(components)
#     chi_sq = compute_interferometric_chi_sq(*compute_observables(components))
#
#     assert chi_sq != 0
#     assert isinstance(chi_sq, float)
#
#     set_data()
#     OPTIONS.model.components = []
#     OPTIONS.model.modulation = 0
#
#
# @pytest.mark.parametrize(
#     "wavelength", [[3.5] * u.um, [8] * u.um, [3.5, 8] * u.um, [3.5, 8, 10] * u.um]
# )
# def test_lnprob(fits_files: List[Path], wavelength: u.um) -> None:
#     """Tests the lnprob function."""
#     static_params = {
#         "dim": 2048,
#         "dist": 145,
#         "pixel_size": 0.1,
#         "eff_temp": 7800,
#         "eff_radius": 1.8,
#         "kappa_abs": 1000,
#         "kappa_cont": 3000,
#     }
#     param_names = ["rin", "p", "c", "s", "sigma0", "weight_cont", "pa", "inc"]
#     values = [1.5, 0.5, 0.3, 0.7, 2000, 0.5, 45, 1.6]
#     limits = [
#         [0, 20],
#         [0, 1],
#         [-1, 1],
#         [-1, 1],
#         [100, 10000],
#         [0, 1],
#         [0, 360],
#         [1, 50],
#     ]
#     params = {name: Parameter(base=name) for name in param_names}
#     for index, param in enumerate(params.values()):
#         param.min, param.max = limits[index]
#         param.value = values[index]
#         param.free = True
#         if param.name in ["p", "pa", "inc"]:
#             param.shared = True
#
#     shared_params = {"p": params["p"], "pa": params["pa"], "inc": params["inc"]}
#     del params["p"]
#     del params["pa"]
#     del params["inc"]
#
#     OPTIONS.model.components = components = [
#         Star(),
#         AsymGreyBody(**params, **shared_params),
#     ]
#     OPTIONS.model.modulation = 1
#
#     set_data(fits_files, wavelengths=wavelength)
#     theta = get_theta(components)
#
#     chi_sq = lnprob(theta)
#     assert isinstance(chi_sq, float)
#     assert chi_sq != 0
#
#     set_data()
#     OPTIONS.model.components = []
#     OPTIONS.model.modulation = 0
