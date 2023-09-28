from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.modeling.models import BlackBody

from ppdmod.component import Component, AnalyticalComponent,\
        NumericalComponent, HankelComponent
from ppdmod.data import ReadoutFits
from ppdmod.parameter import STANDARD_PARAMETERS, Parameter
from ppdmod.options import OPTIONS


COMPONENT_DIR = Path("component")
if not COMPONENT_DIR.exists():
    COMPONENT_DIR.mkdir()

@pytest.fixture
def readout() -> Path:
    """A MATISSE (.fits)-file."""
    file = "hd_142666_2022-04-23T03_05_25:2022-04-23T02_28_06_AQUARIUS_FINAL_TARGET_INT.fits"
    return ReadoutFits(Path("data/fits") / file)


@pytest.fixture
def wavelength() -> u.um:
    """A wavelength grid."""
    return (8.28835527e-06*u.m).to(u.um)


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


@pytest.fixture
def hankel_component() -> HankelComponent:
    """Initializes a numerical component."""
    hankel_component = HankelComponent(
            rin=0.5, rout=3, pa=33,
            elong=0.5, dim=512, a=0.5,
            inner_temp=1500, q=0.5)
    hankel_component.optically_thick = True
    return hankel_component


def test_component(component: Component) -> None:
    """Tests if the initialization of the component works."""
    assert not component.params["pa"].free and\
            not component.params["elong"].free
    component.elliptic = True
    assert component.params["pa"].free and\
            component.params["elong"].free
    assert len(component.params) == 6
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


# TODO: Make image of both radius to infinity and rotated with finite radius.
def test_radius_calculation(component: Component) -> None:
    """Tests if the radius calculated from the grid works."""
    dim, pixel_size = 512, 0.1*u.mas
    grid = component._calculate_internal_grid(dim, pixel_size)
    plt.imshow(np.hypot(*grid))
    plt.title("Image space")
    plt.xlabel("dim [px]")
    plt.savefig(COMPONENT_DIR / "grid.pdf", format="pdf")
    plt.close()

    radius = np.hypot(*grid)
    radial_profile = np.logical_and(radius > 2, radius < 10)
    plt.imshow(radius*radial_profile)
    plt.title("Image space")
    plt.xlabel("dim [px]")
    plt.savefig(COMPONENT_DIR / "grid_ring.pdf", format="pdf")
    plt.close()

    elliptical_component = Component(pa=33, elong=0.6)
    elliptical_component.elliptic = True
    assert elliptical_component.elliptic
    assert elliptical_component.params["pa"]() == 33*u.deg
    assert elliptical_component.params["elong"]() == 0.6*u.one

    grid = elliptical_component._calculate_internal_grid(dim, pixel_size)
    plt.imshow(np.hypot(*grid))
    plt.title("Image space")
    plt.xlabel("dim [px]")
    plt.savefig(COMPONENT_DIR / "elliptic_grid.pdf", format="pdf")
    plt.close()

    radius = np.hypot(*grid)
    radial_profile = np.logical_and(radius > 2, radius < 10)
    plt.imshow(radius*radial_profile)
    plt.title("Image space")
    plt.xlabel("dim [px]")
    plt.savefig(COMPONENT_DIR / "elliptic_grid_ring.pdf", format="pdf")
    plt.close()


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
    assert analytic_component.calculate_image(512, 0.1*u.mas).size > 0


def test_analytic_component_calculate_image_function(
        analytic_component: AnalyticalComponent) -> None:
    """Tests if the visibility function returns None."""
    assert analytic_component._image_function(None, None, None) is None


def test_analytic_component_calculate_visibility_function(
        analytic_component: AnalyticalComponent) -> None:
    """Tests if the visibility function returns None."""
    assert analytic_component._visibility_function(None, None, None) is None


def test_analytical_component_calculate_image(
        analytic_component: AnalyticalComponent) -> None:
    """Tests the analytical component's image calculation."""
    assert analytic_component.calculate_image(512, 0.1*u.mas) is None


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
        numerical_component.calculate_complex_visibility(wavelength=8*u.um)


@pytest.mark.parametrize("grid_type", ["linear", "logarithmic"])
def test_hankel_component_calculate_grid(
        hankel_component: HankelComponent, grid_type: str) -> None:
    """Tests the hankel component's grid calculation."""
    OPTIONS["model.gridtype"] = grid_type
    radius = hankel_component._calculate_internal_grid(512)
    assert radius.unit == u.mas
    assert radius.shape == (512, )
    assert radius[0].value == hankel_component.params["rin"].value\
            and radius[-1].value == hankel_component.params["rout"].value


def test_hankel_component_brightness_function():
    ...


def test_hankel_component_total_flux(
        hankel_component: HankelComponent, wavelength: u.um) -> None:
    """Tests the calculation of the total flux."""
    total_flux = hankel_component.calculate_total_flux(wavelength)
    assert total_flux.unit == u.Jy


@pytest.mark.parametrize("order", [0, 1, 2, 3])
def test_hankel_component_hankel_transform(
        hankel_component: HankelComponent, order: int,
        readout: ReadoutFits, wavelength: u.um) -> None:
    """Tests the hankel component's hankel transformation."""
    radius = hankel_component._calculate_internal_grid(512)
    temp_profile = 1500*u.K*(radius/(hankel_component.params["rin"]()))**(-0.5)
    brightness_profile = BlackBody(temp_profile)(wavelength)
    
    OPTIONS["model.modulation.order"] = order
    visibilities, modulations = hankel_component.hankel_transform(
            brightness_profile.to(u.erg/(u.Hz*u.cm**2*u.s*u.rad**2)),
            radius, readout.ucoord, readout.vcoord, wavelength)
    assert visibilities.shape == (6, )
    if order == 0:
        assert modulations.shape == (0, )
    else:
        assert modulations.shape == (order, 6)
    OPTIONS["model.modulation.order"] = 0


@pytest.mark.parametrize("order", [0, 1, 2, 3])
def test_hankel_component_visibilities(
        hankel_component: HankelComponent, order: int,
        readout: ReadoutFits, wavelength: u.um) -> None:
    """Tests the hankel component's hankel transformation."""
    OPTIONS["model.modulation.order"] = order
    visibilities = hankel_component.calculate_visibilities(
            readout.ucoord, readout.vcoord, wavelength)
    assert visibilities.shape == (6, )
    OPTIONS["model.modulation.order"] = 0


@pytest.mark.parametrize("order", [0, 1, 2, 3])
def test_hankel_component_closure_phases(
        hankel_component: HankelComponent, order: int,
        readout: ReadoutFits, wavelength: u.um) -> None:
    """Tests the hankel component's hankel transformation."""
    OPTIONS["model.modulation.order"] = order
    closure_phases = hankel_component.calculate_closure_phases(
            readout.u123coord, readout.v123coord, wavelength)
    assert closure_phases.shape == (4, )
    OPTIONS["model.modulation.order"] = 0
