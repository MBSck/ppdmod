from pathlib import Path
from typing import List

import astropy.units as u
import numpy as np
import pytest

from ppdmod.component import Component, Convolver
from ppdmod.basic_components import Star, Ring, Gaussian
from ppdmod.data import ReadoutFits
from ppdmod.options import STANDARD_PARAMETERS, OPTIONS
from ppdmod.parameter import Parameter
from ppdmod.data import set_data


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


@pytest.fixture
def convolver(ring: Ring, gaussian: Gaussian) -> Convolver:
    """Initializes a convolver component."""
    return Convolver(ring=ring, gauss=gaussian)


def test_component(component: Component) -> None:
    """Tests if the initialization of the component works."""
    assert not component.pa.free and not component.inc.free

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
    assert params and not params_free


# TODO: Needs better test
def test_translate_coordinates(component: Component) -> None:
    """Tests if the translation of the coordinates works."""
    component.x.value = 10*u.mas
    component.y.value = 10*u.mas
    assert component.translate_image_func(10, 10) == (0*u.mas, 0*u.mas)
    assert component.translate_image_func(0, 0) == (-10*u.mas, -10*u.mas)


# TODO: There is a weird 90 degree rotation here 
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
    set_data([fits_file], wavelengths=wavelength, fit_data=["vis", "t3"])
    fluxes, position = [2, 8]*u.Jy, [5, 0]*u.mas

    vis = OPTIONS.data.vis
    component_one = Star(f=fluxes[0], x=0, y=0)
    component_two = Star(f=fluxes[1], x=position[1], y=position[0], pa=pos_angle)
    vis_one = component_one.compute_complex_vis(vis.ucoord, vis.vcoord, wavelength)
    vis_two = component_two.compute_complex_vis(vis.ucoord, vis.vcoord, wavelength)
    vis_combined = np.abs(vis_one + vis_two)/fluxes.value.sum()

    assert np.allclose(vis_combined, vis.value, atol=1e-3)

    t3 = OPTIONS.data.t3
    t3_one = component_one.compute_complex_vis(t3.u123coord, t3.v123coord, wavelength)
    t3_two = component_two.compute_complex_vis(t3.u123coord, t3.v123coord, wavelength)
    t3_combined = np.angle(np.prod(t3_one + t3_two, axis=1), deg=True)

    assert np.allclose(t3_combined, t3.value, atol=1e-1)
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


def test_convolver_init(convolver: Convolver) -> None:
    """Tests the convolutor's initialization."""
    assert "ring" in vars(convolver).keys()
    assert "gauss" in vars(convolver).keys()
    assert isinstance(convolver.ring, Ring)
    assert isinstance(convolver.gauss, Gaussian)


def test_convolver_components(convolver: Convolver) -> None:
    """Tests the convolutor's components."""
    assert len(convolver.components) == 2
    assert "ring" in convolver.components and "gauss" in convolver.components


# TODO: Do tests for different wavelengths
def test_convolver_vis_func(convolver: Convolver, wavelength: u.um,
                            readout: ReadoutFits) -> None:
    """Tests the convolutor's vis function."""
    vis = convolver.compute_complex_vis(readout.vis2.ucoord, readout.vis2.vcoord, wavelength)
    assert vis.shape == (wavelength.size, 6)
    assert isinstance(vis, np.ndarray)

    t3 = convolver.compute_complex_vis(readout.t3.u123coord, readout.t3.v123coord, wavelength)
    assert t3.shape == (wavelength.size, 3, 4)
    assert isinstance(vis, np.ndarray)
