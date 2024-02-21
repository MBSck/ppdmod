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
    assert not component.pa.free and\
            not component.elong.free
    component.elliptic = True
    assert component.pa.free and\
            component.elong.free
    assert component.x() == 0*u.mas
    assert component.y() == 0*u.mas
    assert component.dim() == 128


def test_eval(component: Component) -> None:
    """Tests if the evaulation of the parameters works."""
    x = Parameter(**STANDARD_PARAMETERS.fov)
    params = {"x": x, "y": 10, "dim": 512}
    component.eval(**params)

    assert component.x == Parameter(**STANDARD_PARAMETERS.fov)
    assert component.y() == 10*u.mas
    assert component.dim() == 512


# TODO: Needs better test
def test_translate_coordinates(component: Component) -> None:
    """Tests if the translation of the coordinates works."""
    assert component.translate_image_func(0, 0) == (0*u.mas, 0*u.mas)
    assert component.translate_image_func(25, 15) == (25*u.mas, 15*u.mas)

# TODO: Why is there a 90 degree turn in aspro? -> Check
@pytest.mark.parametrize(
        "fits_file, pos_angle",
        [("bin_sep.fits", 0*u.deg),
         ("bin_neg_sep.fits", -180*u.deg),
         ("bin_sep_rot90.fits", 90*u.deg),
         ("bin_neg_sep_rot90.fits", -90*u.deg),
         ("bin_sep_rot45.fits", 45*u.deg),
         ("bin_neg_sep_rot45.fits", -45*u.deg),
         ("bin_sep_rot33.fits", 33*u.deg)])
def test_translate_fourier(
        fits_file: List[Path], 
        pos_angle: u.mas, wavelength: u.um) -> None:
    """Tests if the translation of the fourier transform works."""
    fits_file = Path("data/aspro") / fits_file
    set_fit_wavelengths(wavelength)
    set_data([fits_file], fit_data=["vis", "t3"])
    fluxes = [2, 8]*u.Jy

    position = [5, 0]*u.mas
    x = position[0]*np.cos(pos_angle) - position[1]*np.sin(pos_angle)
    y = position[0]*np.sin(pos_angle) + position[1]*np.cos(pos_angle)
    position = [x, y]*u.mas

    vis = OPTIONS.data.vis
    component_one = Star(f=fluxes[0], x=0, y=0)
    component_two = Star(f=fluxes[1], x=position[1], y=position[0])
    vis_one = component_one.compute_vis(vis.ucoord, vis.vcoord, wavelength)
    vis_two = component_two.compute_vis(vis.ucoord, vis.vcoord, wavelength)
    vis_combined = np.abs(vis_one + vis_two)/fluxes.value.sum()

    assert np.allclose(vis_combined, vis.value, atol=1e-3)

    t3 = OPTIONS.data.t3
    t3_one = component_one.compute_vis(t3.u123coord, t3.v123coord, wavelength)
    t3_two = component_two.compute_vis(t3.u123coord, t3.v123coord, wavelength)
    t3_combined = np.angle(np.prod(t3_one + t3_two, axis=1), deg=True)

    assert np.allclose(t3_combined, t3.value, atol=1e-1)


def test_flux_func() -> None:
    ...


def test_vis_func() -> None:
    ...


def test_compute_flux() -> None:
    ...


def test_compute_vis() -> None:
    ...


def test_image_func() -> None:
    ...


def test_compute_image() -> None:
    ...
