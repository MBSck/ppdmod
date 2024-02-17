from pathlib import Path
from typing import List

import astropy.units as u
import numpy as np
import pytest

from ppdmod.component import Component
from ppdmod.custom_components import Star
from ppdmod.options import STANDARD_PARAMETERS, OPTIONS
from ppdmod.parameter import Parameter
from ppdmod.utils import binary_vis
from ppdmod.data import set_data, set_fit_wavelengths


# TODO: Test hankel for multiple wavelengths
@pytest.fixture
def wavelength() -> u.um:
    """A wavelength grid."""
    return [10]*u.um


@pytest.fixture
def component() -> Component:
    """Initializes a component."""
    return Component(pixel_size=0.1)


@pytest.fixture
def fits_files() -> Path:
    """MATISSE (.fits)-files."""
    return list(Path("data/fits").glob("*2022-04-23*.fits"))


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


def test_translate_coordinates(component: Component) -> None:
    """Tests if the translation of the coordinates works."""
    assert component.translate_image_func(0, 0) == (0*u.mas, 0*u.mas)
    assert component.translate_image_func(25, 15) == (25*u.mas, 15*u.mas)

# TODO: Think of better way to do this test. Right now kinda testing itself
@pytest.mark.parametrize("positions_one, positions_two",
                         [[[0, -5]*u.mas, [0, 5]*u.mas],
                          [[-5, 0]*u.mas, [5, 0]*u.mas],
                          [[-5, -5]*u.mas, [5, 5]*u.mas],
                          [[-5, -3.5]*u.mas, [5, 5]*u.mas]])
def test_translate_fourier(fits_files: List[Path],
                           positions_one: u.mas,
                           positions_two: u.mas,
                           wavelength: u.um) -> None:
    """Tests if the translation of the fourier transform works."""
    set_fit_wavelengths(wavelength)
    set_data([fits_files[1]], fit_data=["vis", "t3"])
    atol, fluxes = 1e-6, [3, 9]*u.Jy

    vis = OPTIONS.data.vis
    component_one = Star(f=fluxes[0], x=positions_one[0], y=positions_one[1])
    component_two = Star(f=fluxes[1], x=positions_two[0], y=positions_two[1])
    vis_one = component_one.compute_vis(vis.ucoord, vis.vcoord, wavelength)
    vis_two = component_two.compute_vis(vis.ucoord, vis.vcoord, wavelength)
    vis_combined = vis_one + vis_two
    vis_binary = binary_vis(
            fluxes[0], fluxes[1], vis.ucoord, vis.vcoord,
            positions_one, positions_two, wavelength=wavelength)

    assert np.allclose(vis_combined, vis_binary, atol=atol)
    assert np.allclose(np.abs(vis_combined), np.abs(vis_binary), atol=atol)
    assert np.allclose(np.angle(vis_combined), np.angle(vis_binary), atol=atol)

    t3 = OPTIONS.data.t3
    t3_one = component_one.compute_vis(t3.u123coord, t3.v123coord, wavelength)
    t3_two = component_two.compute_vis(t3.u123coord, t3.v123coord, wavelength)
    t3_combined = t3_one + t3_two
    t3_binary = binary_vis(
            fluxes[0], fluxes[1], t3.u123coord, t3.v123coord,
            position1=positions_one, position2=positions_two,
            wavelength=wavelength)

    assert np.allclose(t3_combined, t3_binary, atol=atol)
    assert np.allclose(np.abs(t3_combined), np.abs(t3_binary), atol=atol)
    assert np.allclose(np.angle(t3_combined), np.angle(t3_binary), atol=atol)



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
