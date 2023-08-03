import astropy.units as u
import numpy as np
import pytest

from ppdmod.component import Component, AnalyticalComponent, NumericalComponent
from ppdmod.parameter import STANDARD_PARAMETERS, Parameter


@pytest.fixture
def component() -> Component:
    """Initializes a component."""
    return Component()


@pytest.fixture
def analytic_component() -> AnalyticalComponent:
    """Initializes an analytical component."""
    return AnalyticalComponent()


@pytest.fixture
def numerical_component() -> NumericalComponent:
    """Initializes a numerical component."""
    return NumericalComponent()


def test_component(component: Component) -> None:
    """Tests if the initialization of the component works."""
    assert len(component.params) == 3
    assert component.params["x"]() == 0*u.mas
    assert component.params["y"]() == 0*u.mas
    assert component.params["dim"]() == 128


def test_eval(component: Component) -> None:
    """Tests if the evaulation of the parameters works."""
    x = Parameter(**STANDARD_PARAMETERS["fov"])
    params = {"x": x, "y": 10, "dim": 512}
    component._eval(**params)

    assert component.params["x"] == Parameter(**STANDARD_PARAMETERS["fov"])
    assert component.params["y"]() == 10*u.mas
    assert component.params["dim"]() == 512


def test_translate_fourier(component: Component) -> None:
    """Tests if the translation of the fourier transform works."""
    assert component._translate_fourier_transform(0, 0) == 1
    assert component._translate_fourier_transform(25, 15) == 1


def test_translate_coordinates(component: Component) -> None:
    """Tests if the translation of the coordinates works."""
    assert component._translate_coordinates(0, 0) == (0, 0)
    assert component._translate_coordinates(25, 15) == (25, 15)


def test_analytic_component_init(analytic_component: AnalyticalComponent) -> None:
    """Tests if the initialization of the analytical component works."""
    analytic_component.elliptic = True
    analytic_component.__init__()
    def image_function(xx, yy, wl):
        return np.hypot(xx, yy)*np.inf
    analytic_component._image_function = image_function

    assert "pa" in analytic_component.params
    assert "elong" in analytic_component.params
    assert analytic_component.calculate_image(512).size > 0


def test_analytic_component_image_function(
        analytic_component: AnalyticalComponent) -> None:
    """Tests if the visibility function returns None."""
    assert analytic_component._image_function(None, None) is None


def test_analytic_component_visibility_function(
        analytic_component: AnalyticalComponent) -> None:
    """Tests if the visibility function returns None."""
    assert analytic_component._visibility_function() is None


def test_numerical_component_init(numerical_component: NumericalComponent) -> None:
    """Tests if the initialization of the numerical component works."""
    numerical_component.elliptic = True
    numerical_component.__init__()
    assert "pa" in numerical_component.params
    assert "elong" in numerical_component.params
    assert "pixel_size" in numerical_component.params


def test_numerical_component_grid(numerical_component: NumericalComponent) -> None:
    """Tests the grid calculation of the numerical component."""
    numerical_component.params["pixel_size"].value = 0.1
    grid = numerical_component._calculate_internal_grid()
    dims = (numerical_component.params["dim"](),
            numerical_component.params["dim"]())
    assert all(axe.shape == dims for axe in grid)
    assert all(axe.unit == u.mas for axe in grid)
    assert all(0.*u.mas in axe for axe in grid)
