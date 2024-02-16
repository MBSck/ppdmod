import astropy.units as u
import pytest

from ppdmod.component import Component
from ppdmod.options import STANDARD_PARAMETERS
from ppdmod.parameter import Parameter


# TODO: Test hankel for multiple wavelengths
@pytest.fixture
def wavelength() -> u.um:
    """A wavelength grid."""
    return [12.5]*u.um


@pytest.fixture
def component() -> Component:
    """Initializes a component."""
    return Component(pixel_size=0.1)


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


def test_translate_fourier(component: Component) -> None:
    """Tests if the translation of the fourier transform works."""
    assert component.translate_fourier_space(0, 0, 8*u.um) == 1
    assert component.translate_fourier_space(25, 15, 8*u.um) == 1


def test_translate_coordinates(component: Component) -> None:
    """Tests if the translation of the coordinates works."""
    assert component.translate_coordinates(0, 0) == (0*u.mas, 0*u.mas)
    assert component.translate_coordinates(25, 15) == (25*u.mas, 15*u.mas)


# TODO: Write test for compute_vis and compute_t3 and all compute functions
def test_flux_func() -> None:
    ...


def test_vis_func() -> None:
    ...


def test_t3_func() -> None:
    ...


def test_compute_flux() -> None:
    ...


def test_compute_vis() -> None:
    ...


def test_compute_t3() -> None:
    ...


def test_image_func() -> None:
    ...


def test_compute_image() -> None:
    ...
