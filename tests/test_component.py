from pathlib import Path
from typing import List

import astropy.units as u
import pytest

from ppdmod.basic_components import Ring
from ppdmod.component import Component, FourierComponent
from ppdmod.data import ReadoutFits
from ppdmod.options import OPTIONS
from ppdmod.parameter import Parameter


@pytest.fixture
def readout() -> ReadoutFits:
    """Initializes the readout."""
    return ReadoutFits(list(Path("data/fits").glob("*2022-04-23*.fits"))[0])


@pytest.fixture
def wavelength() -> u.um:
    """A wavelength grid."""
    return [10] * u.um


@pytest.fixture
def component() -> FourierComponent:
    """Initializes a component."""
    return FourierComponent()


@pytest.fixture
def fits_files() -> List[Path]:
    """MATISSE (.fits)-files."""
    return list(Path("data/fits").glob("*2022-04-23*.fits"))


@pytest.fixture
def ring() -> Ring:
    """Initializes a gaussian component."""
    return Ring(dim=512, rin=5, width=1)


def test_component(component: Component) -> None:
    assert hasattr(component, "x")
    assert hasattr(component, "y")
    assert hasattr(component, "dim")


def test_eval(component: Component) -> None:
    """Tests if the evaulation of the parameters works."""
    component.eval(**{"x": Parameter(base="fov"), "y": 10, "dim": 512})

    assert component.x == Parameter(base="fov")
    assert component.y() == 10 * u.mas
    assert component.dim.value == 512


def test_get_params(component: Component):
    """Tests the fetching of params from a component."""
    assert component.get_params()
    assert not component.get_params(free=True)
    assert not component.get_params(shared=True)

    component.x.free = True
    component.y.free = component.y.shared = True
    assert component.get_params(free=True)
    assert "x" in component.get_params(free=True).keys()
    assert "y" not in component.get_params(free=True).keys()
    assert component.get_params(shared=True)
    assert "y" in component.get_params(shared=True).keys()


def test_fourier_component(component: FourierComponent) -> None:
    """Tests if the initialization of the component works."""
    assert not (component.pa.free and component.inc.free)

    component.elliptic = True
    assert component.pa.free and component.inc.free

    component.asymmetric = True
    assert hasattr(component, "c1")
    assert hasattr(component, "s1")

    OPTIONS.model.modulation = 3
    component.__init__()
    for i in range(1, OPTIONS.model.modulation + 1):
        assert hasattr(component, f"c{i}")
        assert hasattr(component, f"s{i}")

    OPTIONS.model.modulation = 1


# TODO: Needs better test
def test_translate_coordinates(component: FourierComponent) -> None:
    """Tests if the translation of the coordinates works."""
    component.x.value = 10 * u.mas
    component.y.value = 10 * u.mas
    assert component.translate_image_func(10, 10) == (0 * u.mas, 0 * u.mas)
    assert component.translate_image_func(0, 0) == (-10 * u.mas, -10 * u.mas)


def test_copy(component: FourierComponent) -> None: ...


def test_flux_func() -> None: ...


def test_vis_func() -> None: ...


def test_compute_flux() -> None: ...


def test_compute_vis() -> None: ...


def test_image_func() -> None: ...


def test_compute_image() -> None: ...
