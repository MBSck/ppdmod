from pathlib import Path
from typing import Dict

import astropy.units as u
import numpy as np
import pytest

from ppdmod.custom_components import Star, TemperatureGradient
from ppdmod.model import Model
from ppdmod.readout import ReadoutFits
from ppdmod.parameter import Parameter
from ppdmod.options import OPTIONS
from ppdmod.utils import get_binned_dimension, linearly_combine_opacities


@pytest.fixture
def star_parameters() -> Dict[str, float]:
    """The star's parameters"""
    return {"dist": 145, "eff_temp": 7800, "eff_radius": 1.8}


@pytest.fixture
def qval_file_dir() -> Path:
    """The qval-file directory."""
    return Path("/Users/scheuck/Data/opacities/QVAL")


@pytest.fixture
def wavelength_solution() -> u.um:
    """A MATISSE (.fits)-file."""
    path = Path("/Users/scheuck/Data/reduced_data/hd142666/matisse")
    file = Path("hd_142666_2022-04-23T03_05_25:2022-04-23T02_28_06_AQUARIUS_FINAL_TARGET_INT.fits")
    return (ReadoutFits(path / file).wavelength*u.m).to(u.um)


@pytest.fixture
def opacity(qval_file_dir: Path,
            wavelength_solution: u.um) -> None:
    """A parameter containing the opacity."""
    weights = np.array([42.8, 9.7, 43.5, 1.1, 2.3, 0.6])/100
    qval_files = ["Q_Am_Mgolivine_Jae_DHS_f1.0_rv0.1.dat",
                  "Q_Am_Mgolivine_Jae_DHS_f1.0_rv1.5.dat",
                  "Q_Am_Mgpyroxene_Dor_DHS_f1.0_rv1.5.dat",
                  "Q_Fo_Suto_DHS_f1.0_rv0.1.dat",
                  "Q_Fo_Suto_DHS_f1.0_rv1.5.dat",
                  "Q_En_Jaeger_DHS_f1.0_rv1.5.dat"]
    qval_paths = list(map(lambda x: qval_file_dir / x, qval_files))
    opacity = linearly_combine_opacities(weights,
                                         qval_paths, wavelength_solution)
    return Parameter(name="kappa_abs", value=opacity,
                     wavelength=wavelength_solution,
                     unit=u.cm**2/u.g, free=False,
                     description="Dust mass absorption coefficient")


@pytest.fixture
def temp_gradient_parameters() -> Dict[str, float]:
    """The temperature gradient's parameters."""
    return {"rin": 0.5, "rout": 100, "dust_mass": 0.11, "q": 0.5,
            "inner_temp": 1500, "pixel_size": 0.1, "p": 0.5}


@pytest.fixture
def star(star_parameters: Dict[str, float]) -> Star:
    """Initializes a star component."""
    return Star(**star_parameters)


@pytest.fixture
def temp_gradient(star_parameters: Dict[str, float],
                  temp_gradient_parameters: Dict[str, float]
                  ) -> TemperatureGradient:
    """Initializes a temperature gradient component."""
    return TemperatureGradient(**star_parameters,
                               **temp_gradient_parameters)


def test_model_init(star: Star, temp_gradient: TemperatureGradient) -> None:
    """Tests the model's initialization."""
    model = Model(star, temp_gradient)
    model_list = Model([star, temp_gradient])
    model_list_one_comp = Model([star])
    assert model.components == (star, temp_gradient,)
    assert model_list.components == (star, temp_gradient,)
    assert model_list_one_comp.components == (star,)


def test_model_params(star: Star, temp_gradient: TemperatureGradient) -> None:
    """Tests the model's parameters property."""
    model_star, model_temp_gradient = Model(star), Model(temp_gradient)
    star_expected = ["x", "y", "dim", "eff_temp", "eff_radius", "dist"]
    temp_gradient_expected = ["x", "y", "dim", "eff_temp",
                              "eff_radius", "dist", "inner_temp", "pa",
                              "elong", "p", "q", "dust_mass", "kappa_abs",
                              "rin", "rout"]
    param_star_names = [param.name for param in model_star.params.values()]
    param_temp_gradient_names = [param.name for
                                 param in model_temp_gradient.params.values()]
    assert all(param in param_star_names for param in star_expected)
    assert all(param in param_temp_gradient_names
               for param in temp_gradient_expected)


def test_model_free_params(star: Star,
                           temp_gradient: TemperatureGradient) -> None:
    """Tests the model's parameters property."""
    model_star, model_temp_gradient = Model(star), Model(temp_gradient)
    temp_gradient_expected = ["pa", "elong", "p", "q", "rin", "rout"]
    param_temp_gradient_names = [param.name for
                                 param in model_temp_gradient.params.values()]
    assert not model_star.free_params
    assert all(param in param_temp_gradient_names
               for param in temp_gradient_expected)


# def test_calculate_image(star: Star,
#                          temp_gradient: TemperatureGradient,
#                          opacity: Parameter) -> None:
#     """Tests the model's image calculation."""
#     temp_gradient.params["kappa_abs"] = opacity
#     breakpoint()
#     model = Model(star, temp_gradient)
#     image = model.calculate_image(512, 0.1, 8*u.um)
#     assert image.unit == u.Jy
#     assert image.shape == (512, 512)
#     assert image[256, 256] != 0*u.Jy


# def test_calculate_complex_visibility(
#         star: Star, temp_gradient: TemperatureGradient) -> None:
#     """Tests the model's complex visibility function calculation."""
#     model = Model(star, temp_gradient)
#     complex_vis = model.calculate_complex_visibility(8*u.um)
#     binned_dim = get_binned_dimension(star.params["dim"](),
#                                       OPTIONS["fourier.binning"])
#     assert complex_vis.unit == u.one
#     assert complex_vis.shape == (binned_dim, binned_dim)
