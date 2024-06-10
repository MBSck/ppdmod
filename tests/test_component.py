from pathlib import Path
from typing import List

import astropy.units as u
import numpy as np
import pytest

from ppdmod.component import Component
from ppdmod.basic_components import Star, Ring, Gaussian
from ppdmod.data import ReadoutFits, set_data
from ppdmod.options import STANDARD_PARAMETERS, OPTIONS
from ppdmod.parameter import Parameter
from ppdmod.utils import compute_vis, compute_t3


@pytest.fixture
def readout() -> ReadoutFits:
    """Initializes the readout."""
    return ReadoutFits(list(Path("data/fits").glob("*2022-04-23*.fits"))[0])


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


@pytest.fixture
def ring() -> Ring:
    """Initializes a gaussian component."""
    return Ring(**{"dim": 512, "diam": 5, "width": 1})


@pytest.fixture
def gaussian() -> Gaussian:
    """Initializes a gaussian component."""
    return Gaussian(**{"dim": 512, "fwhm": 0.5})


def test_component(component: Component) -> None:
    """Tests if the initialization of the component works."""
    assert component.pa.free and component.inc.free

    component.elliptic = True
    assert component.pa.free and component.inc.free
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


# TODO: Finish this test
def test_get_params(component: Component):
    """Tests the fetching of params from a component."""
    params = component.get_params()
    params_free = component.get_params(free=True)
    assert params and params_free


# TODO: Needs better test
def test_translate_coordinates(component: Component) -> None:
    """Tests if the translation of the coordinates works."""
    component.x.value = 10*u.mas
    component.y.value = 10*u.mas
    assert component.translate_image_func(10, 10) == (0*u.mas, 0*u.mas)
    assert component.translate_image_func(0, 0) == (-10*u.mas, -10*u.mas)


@pytest.mark.parametrize(
        "fits_file, pos_angle",
        [
        ("bin_sep5_pa0_extended.fits", 0*u.deg),
         ("bin_sep5_pa180_extended.fits", 180*u.deg),
         ("bin_sep5_pa-180_extended.fits", -180*u.deg),
         ("bin_sep5_pa90_extended.fits", 90*u.deg),
         ("bin_sep5_pa-90_extended.fits", -90*u.deg),
         ("bin_sep5_pa45_extended.fits", 45*u.deg),
         ("bin_sep5_pa-45_extended.fits", -45*u.deg),
         ("bin_sep5_pa33_extended.fits", 33*u.deg)
        ])
def test_translate_fourier(
        fits_file: List[Path], pos_angle: u.mas) -> None:
    """Tests if the translation of the fourier transform works."""
    fits_file = Path("data/aspro") / fits_file
    wavelength = [3]*u.um
    data = set_data([fits_file], wavelengths=wavelength, fit_data=["vis", "t3"])
    fluxes = [2, 8]*u.Jy

    vis = data.vis
    component_one = Star(f=fluxes[0], x=0, y=0)
    component_two = Star(f=fluxes[1], x=0, y=5, pa=pos_angle)
    vis_one = component_one.compute_complex_vis(vis.ucoord, vis.vcoord, wavelength)
    vis_two = component_two.compute_complex_vis(vis.ucoord, vis.vcoord, wavelength)
    vis_combined = compute_vis(vis_one + vis_two)
    vis_combined /= vis_combined[:, 0]

    assert np.allclose(vis_combined[:, 1:], vis.value, atol=1e-2)

    t3 = data.t3
    t3_one = component_one.compute_complex_vis(t3.u123coord, t3.v123coord, wavelength)
    t3_two = component_two.compute_complex_vis(t3.u123coord, t3.v123coord, wavelength)
    t3_combined = compute_t3(t3_one + t3_two)

    diff = np.ptp(np.hstack(
        (t3.value[0][:, np.newaxis], t3_combined[:, 1:][0][:, np.newaxis])), axis=1)
    assert diff.max() < 1
    set_data(fit_data=["vis", "t3"])


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
