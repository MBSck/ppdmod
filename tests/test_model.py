from typing import Dict

import astropy.units as u
import pytest

from ppdmod.custom_components import Star, TemperatureGradient
from ppdmod.model import Model
from ppdmod.options import OPTIONS
from ppdmod.utils import get_binned_dimension


@pytest.fixture
def star_parameters() -> Dict[str, float]:
    """The star's parameters"""
    return {"dist": 145, "eff_temp": 7800, "eff_radius": 1.8}


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


def test_calculate_image(star: Star,
                         temp_gradient: TemperatureGradient) -> None:
    """Tests the model's image calculation."""
    model = Model(star, temp_gradient)
    image = model.calculate_image(512, 0.1, 8*u.um)
    assert image.unit == u.Jy
    assert image.shape == (512, 512)


def test_calculate_complex_visibility(
    star: Star, temp_gradient: TemperatureGradient) -> None:
    """Tests the model's complex visibility function calculation."""
    model = Model(star, temp_gradient)
    complex_vis = model.calculate_complex_visibility(8*u.um)
    binned_dim = get_binned_dimension(star.params["idim"](), OPTIONS["fourier.binning"])
    assert complex_vis.unit == u.one
    assert complex_vis.shape == (binned_dim, binned_dim)
