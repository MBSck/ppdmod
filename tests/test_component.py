from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ppdmod.component import Component, AnalyticalComponent, NumericalComponent
from ppdmod.parameter import STANDARD_PARAMETERS, Parameter


@pytest.fixture
def component() -> Component:
    """Initializes a component."""
    return Component(pixel_size=0.1)


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
    assert len(component.params) == 4
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


def test_component_calculate_grid(component: Component) -> None:
    """Tests the grid calculation of the numerical component.

    Also tests if the phase space is correctly calculated by
    making pictures to check it.
    """
    grid = component._calculate_internal_grid()
    dims = (component.params["dim"](),
            component.params["dim"]())
    assert all(axe.shape == dims for axe in grid)
    assert all(axe.unit == u.mas for axe in grid)
    assert all(0.*u.mas in axe for axe in grid)

    dim, pixel_size = 512, 0.1
    grid = component._calculate_internal_grid(dim, pixel_size)
    assert all(axe.shape == (512, 512) for axe in grid)
    assert all(axe.unit == u.mas for axe in grid)
    assert all(0.*u.mas in axe for axe in grid)

    # NOTE: Checks the phase space.
    radius = np.hypot(*grid)
    radius[radius > 4*u.mas] = 0*u.mas
    radius[radius != 0*u.mas] = 1*u.mas
    ft = compute_2Dfourier_transform(radius.value)

    phase_dir = Path("phase")
    if not phase_dir.exists():
        phase_dir.mkdir()
    _, (ax, bx, cx) = plt.subplots(1, 3)

    ax.imshow(radius.value)
    ax.set_title("Magnitude")
    ax.set_xlabel("dim [px]")
    bx.imshow(np.abs(ft))
    bx.set_title("Magnitude")
    bx.set_xlabel("dim [px]")
    cx.imshow(np.angle(ft))
    cx.set_title("Phase")
    cx.set_xlabel("dim [px]")
    plt.savefig(phase_dir / f"fourier_space", format="pdf")


# TODO: Make image of both radius to infinity and rotated with finite radius.
def test_radius_calculation(component: Component) -> None:
    """Tests if the radius calculated from the grid works."""
    component_dir = Path("component")
    if not component_dir.exists():
        component_dir.mkdir()

    grid = component._calculate_internal_grid()
    plt.imshow(np.hypot(*grid).value)
    plt.title("Image space")
    plt.xlabel("dim [px]")
    plt.savefig(component_dir / "grid_automatic.pdf", format="pdf")

    dim, pixel_size = 512, 0.1
    plt.imshow(np.hypot(*grid).value)
    plt.title("Image space")
    plt.xlabel("dim [px]")
    plt.savefig(component_dir / "grid_manual.pdf", format="pdf")

    grid = component._calculate_internal_grid(dim, pixel_size)
    radius = np.hypot(*grid).value
    radial_profile = np.logical_and(radius > 2, radius < 10)
    plt.imshow(radius*radial_profile)
    plt.title("Image space")
    plt.xlabel("dim [px]")
    plt.savefig(component_dir / "grid_manual_ring.pdf", format="pdf")

    elliptical_component = Component(pa=33, elong=0.6)
    assert elliptical_component.elliptic
    assert elliptical_component.params["pa"]() == 33*u.deg
    assert elliptical_component.params["elong"]() == 0.6*u.one

    grid = elliptical_component._calculate_internal_grid()
    plt.imshow(np.hypot(*grid).value)
    plt.title("Image space")
    plt.xlabel("dim [px]")
    plt.savefig(component_dir / "elliptic_grid_automatic.pdf", format="pdf")

    grid = elliptical_component._calculate_internal_grid(dim, pixel_size)
    plt.imshow(np.hypot(*grid).value)
    plt.title("Image space")
    plt.xlabel("dim [px]")
    plt.savefig(component_dir / "elliptic_grid_manual.pdf", format="pdf")

    grid = elliptical_component._calculate_internal_grid(dim, pixel_size)
    radius = np.hypot(*grid).value
    radial_profile = np.logical_and(radius > 2, radius < 10)
    plt.imshow(radius*radial_profile)
    plt.title("Image space")
    plt.xlabel("dim [px]")
    plt.savefig(component_dir / "elliptic_grid_manual_ring.pdf", format="pdf")


def test_translate_fourier(component: Component) -> None:
    """Tests if the translation of the fourier transform works."""
    assert component._translate_fourier_transform(0, 0, 8*u.um) == 1
    assert component._translate_fourier_transform(25, 15, 8*u.um) == 1


def test_translate_coordinates(component: Component) -> None:
    """Tests if the translation of the coordinates works."""
    assert component._translate_coordinates(0, 0) == (0*u.mas, 0*u.mas)
    assert component._translate_coordinates(25, 15) == (25*u.mas, 15*u.mas)


def test_analytic_component_init(analytic_component: AnalyticalComponent) -> None:
    """Tests if the initialization of the analytical component works."""
    analytic_component.elliptic = True
    analytic_component.__init__()

    def image_function(xx, yy, wl):
        return np.hypot(xx, yy)
    analytic_component._image_function = image_function

    assert "pa" in analytic_component.params
    assert "elong" in analytic_component.params
    assert analytic_component.calculate_image(512).size > 0


def test_analytic_component_calculate_image_function(
        analytic_component: AnalyticalComponent) -> None:
    """Tests if the visibility function returns None."""
    assert analytic_component._image_function(None, None) is None


def test_analytic_component_calculate_visibility_function(
        analytic_component: AnalyticalComponent) -> None:
    """Tests if the visibility function returns None."""
    assert analytic_component._visibility_function() is None


def test_analytical_component_calculate_image(
        analytic_component: AnalyticalComponent) -> None:
    """Tests the analytical component's image calculation."""
    assert analytic_component.calculate_image(512, 0.1) is None


def test_analytical_component_calculate_complex_visibility(
        analytic_component: AnalyticalComponent) -> None:
    """Tests the analytical component's complex visibility
    function calculation."""
    assert analytic_component.calculate_complex_visibility() is None


def test_numerical_component_init(numerical_component: NumericalComponent) -> None:
    """Tests if the initialization of the numerical component works."""
    numerical_component.elliptic = True
    numerical_component.__init__()
    assert "pa" in numerical_component.params
    assert "elong" in numerical_component.params
    assert "pixel_size" in numerical_component.params


def test_numerical_component_calculate_image(
        numerical_component: NumericalComponent) -> None:
    """Tests the numerical component's image calculation."""
    assert numerical_component.calculate_image() is None


def test_numerical_component_calculate_complex_visibility(
        numerical_component: NumericalComponent) -> None:
    """Tests the numerical component's complex visibility
    function calculation."""
    # NOTE: Raises attribute error here as image is not calculated
    # and None has no value attribute.
    with pytest.raises(AttributeError) as e_info:
        numerical_component.calculate_complex_visibility(8*u.um)
