import time
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from astropy.modeling.models import BlackBody

from ppdmod import utils
from ppdmod.component import Component, AnalyticalComponent, \
        NumericalComponent, HankelComponent
from ppdmod.data import ReadoutFits
from ppdmod.parameter import STANDARD_PARAMETERS, Parameter
from ppdmod.options import OPTIONS


CALCULATION_FILE = Path("analytical_calculation.xlsx")
COMPONENT_DIR = Path("component")
VISIBILITY = "Visibility"
CLOSURE_PHASE = "Closure Phase"

READOUT = ReadoutFits(Path("data/fits")
                      / "hd_142666_2022-04-23T03_05_25:2022-04-23T02_28_06_AQUARIUS_FINAL_TARGET_INT_flux_avg_vis.fits")
BASELINES = [f"B {baseline}" for baseline in np.around(np.hypot(READOUT.ucoord, READOUT.vcoord), 0)]
TRIANGLES = [f"T {triangle}" for triangle in np.around(np.hypot(READOUT.u123coord, READOUT.v123coord).max(axis=0), 0)]
TRIANGLES[-1] += ".0"

utils.make_workbook(
    CALCULATION_FILE,
    {
        VISIBILITY: ["Dimension (px)", *BASELINES, "Computation Time (s)"],
        CLOSURE_PHASE: ["Dimension (px)", *TRIANGLES, "Computation Time (s)"],
    })


@pytest.fixture
def wavelength() -> u.um:
    """A wavelength grid."""
    return (13.000458e-6*u.m).to(u.um)


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
            elong=0.5, dim=512, a=0.5, phi=33,
            inner_temp=1500, q=0.5)
    hankel_component.optically_thick = True
    hankel_component.asymmetric = True
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
    total_flux = hankel_component.calculate_flux(wavelength)
    assert total_flux


# TODO: Write here check if higher orders are implemented
@pytest.mark.parametrize("order", [0, 1, 2, 3])
def test_hankel_component_hankel_transform(
        hankel_component: HankelComponent,
        order: int, wavelength: u.um) -> None:
    """Tests the hankel component's hankel transformation."""
    radius = hankel_component._calculate_internal_grid(512)
    temp_profile = 1500*u.K*(radius/(hankel_component.params["rin"]()))**(-0.5)
    brightness_profile = BlackBody(temp_profile)(wavelength)

    OPTIONS["model.modulation.order"] = order
    corr_fluxes, modulations = hankel_component.hankel_transform(
            brightness_profile.to(u.erg/(u.Hz*u.cm**2*u.s*u.rad**2)),
            radius, READOUT.ucoord, READOUT.vcoord, wavelength)
    assert corr_fluxes.shape == (6, )
    assert corr_fluxes.unit == u.Jy
    if order == 0:
        assert modulations.shape == (0, )
    else:
        assert modulations.shape == (order, 6)
    OPTIONS["model.modulation.order"] = 0


@pytest.mark.parametrize("order", [0, 1, 2, 3])
def test_hankel_component_corr_fluxes(
        hankel_component: HankelComponent,
        order: int, wavelength: u.um) -> None:
    """Tests the hankel component's hankel transformation."""
    OPTIONS["model.modulation.order"] = order
    corr_fluxes = hankel_component.calculate_corr_flux(
            READOUT.ucoord, READOUT.vcoord, wavelength)
    assert corr_fluxes.shape == (6, )
    assert isinstance(corr_fluxes, np.ndarray)
    OPTIONS["model.modulation.order"] = 0


@pytest.mark.parametrize("order", [0, 1, 2, 3])
def test_hankel_component_closure_phases(
        hankel_component: HankelComponent,
        order: int, wavelength: u.um) -> None:
    """Tests the hankel component's hankel transformation."""
    OPTIONS["model.modulation.order"] = order
    closure_phases = hankel_component.calculate_closure_phase(
            READOUT.u123coord, READOUT.v123coord, wavelength)

    assert closure_phases.shape == (4, )
    OPTIONS["model.modulation.order"] = 0


@pytest.mark.parametrize(
        "dim", [4096, 2096, 1024, 512, 256, 128, 64, 32])
def test_hankel_resolution(dim: int, wavelength: u.um) -> None:
    """Tests the hankel component's resolution."""
    hankel_component = HankelComponent(
            rin=0.5, rout=3, pa=33,
            elong=0.5, dim=dim, a=0.5, phi=33,
            inner_temp=1500, q=0.5)
    hankel_component.optically_thick = True
    hankel_component.asymmetric = True

    OPTIONS["model.modulation.order"] = 1
    start_time_vis = time.perf_counter()
    visibilities = hankel_component.calculate_corr_flux(
            READOUT.ucoord, READOUT.vcoord, wavelength)
    end_time_vis = time.perf_counter()-start_time_vis

    start_time_cphase = time.perf_counter()
    closure_phases = hankel_component.calculate_closure_phase(
            READOUT.u123coord, READOUT.v123coord, wavelength)
    end_time_cphase = time.perf_counter()-start_time_cphase

    vis_data = {"Dimension (px)": [dim],
                **{baseline: value for baseline, value in zip(BASELINES, visibilities)},
                   "Computation Time (s)": [end_time_vis]}

    cphase_data = {"Dimension (px)": [dim],
                   **{triangle: value for triangle, value in zip(TRIANGLES, closure_phases)},
                   "Computation Time (s)": [end_time_cphase]}

    if CALCULATION_FILE.exists():
        df = pd.read_excel(CALCULATION_FILE, sheet_name=VISIBILITY)
        new_df = pd.DataFrame(vis_data)
        df = pd.concat([df, new_df])
    else:
        df = pd.DataFrame(vis_data)

    with pd.ExcelWriter(CALCULATION_FILE, engine="openpyxl",
                        mode="a", if_sheet_exists="replace") as writer:
        df.to_excel(writer, sheet_name=VISIBILITY, index=False)

    if CALCULATION_FILE.exists():
        df = pd.read_excel(CALCULATION_FILE, sheet_name=CLOSURE_PHASE)
        new_df = pd.DataFrame(cphase_data)
        df = pd.concat([df, new_df])
    else:
        df = pd.DataFrame(cphase_data)

    with pd.ExcelWriter(CALCULATION_FILE, engine="openpyxl",
                        mode="a", if_sheet_exists="replace") as writer:
        df.to_excel(writer, sheet_name=CLOSURE_PHASE, index=False)

    OPTIONS["model.modulation.order"] = 0
