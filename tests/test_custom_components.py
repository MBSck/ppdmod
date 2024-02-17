import time
from pathlib import Path
from typing import Dict

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ppdmod import utils
from ppdmod.component import Component
from ppdmod.custom_components import Star, TempGradient, GreyBody, assemble_components
from ppdmod.data import ReadoutFits
from ppdmod.options import STANDARD_PARAMETERS, OPTIONS
from ppdmod.parameter import Parameter


DIMENSION = [2**power for power in range(9, 13)]
CALCULATION_FILE = Path("analytical_calculation.xlsx")
COMPONENT_DIR = Path("component")

READOUT = ReadoutFits(list(Path("data/fits").glob("*2022-04-23*.fits"))[0])
utils.make_workbook(
    CALCULATION_FILE,
    {
        "Vis": ["Dimension (px)", "Computation Time (s)"],
        "T3": ["Dimension (px)", "Computation Time (s)"],
    })


@pytest.fixture
def wavelength() -> u.m:
    """A wavelenght grid."""
    return [12.5]*u.um


@pytest.fixture
def wavelength_solution() -> u.um:
    """A MATISSE (.fits)-file."""
    file = "hd_142666_2022-04-23T03_05_25:2022-04-23T02_28_06_AQUARIUS_FINAL_TARGET_INT.fits"
    return ReadoutFits(Path("data/fits") / file).wavelength


@pytest.fixture
def qval_file_dir() -> Path:
    """The qval-file directory."""
    return Path("data/qval")


@pytest.fixture
def star_parameters() -> Dict[str, float]:
    """The star's parameters"""
    return {"dim": 512, "dist": 145, "eff_temp": 7800, "eff_radius": 1.8}


@pytest.fixture
def temp_gradient_parameters() -> Dict[str, float]:
    """The temperature gradient's parameters."""
    return {"rin": 0.5, "q": 0.5, 
            "inner_temp": 1500, "inner_sigma": 2000,
            "pixel_size": 0.1, "p": 0.5}


@pytest.fixture
def temp_gradient() -> TempGradient:
    """Initializes a numerical component."""
    temp_grad = TempGradient(
            rin=0.5, rout=3, pa=33, dist=148.3,
            elong=0.5, dim=512, a=0.5, phi=33,
            inner_temp=1500, q=0.5)
    temp_grad.optically_thick = True
    temp_grad.asymmetric = True
    return temp_grad


@pytest.fixture
def star(star_parameters: Dict[str, float]) -> Star:
    """Initializes a star component."""
    return Star(**star_parameters)


def test_get_component() -> Component:
    ...


def test_star_init(star: Star) -> None:
    """Tests the star's initialization."""
    assert "dist" in star.params
    assert "eff_temp" in star.params
    assert "eff_radius" in star.params


def test_star_stellar_radius_angular(star: Star) -> None:
    """Tests the stellar radius conversion to angular radius."""
    assert star.stellar_radius_angular.unit == u.mas


# TODO: Include test for stellar flux with input file as well.
def test_star_flux(star: Star, wavelength: u.um) -> None:
    """Tests the calculation of the total flux."""
    assert star.flux_func(wavelength).shape == (wavelength.size, 1)


def test_star_vis_func(star: Star, wavelength: u.um) -> None:
    """Tests the calculation of the total flux."""
    flux = star.vis_func(READOUT.vis.ucoord, READOUT.vis.vcoord, wavelength)
    assert flux.shape == (wavelength.size, READOUT.vis.ucoord.size)


# TODO: Make this for multiple wavelengths
@pytest.mark.parametrize(
        "wl, dim", [(wl, dim) for dim in DIMENSION for wl in [8, 9, 10, 11]*u.um])
def test_star_image(star: Star, dim: int, wl: u.um,
                    wavelength: u.um) -> None:
    """Tests the star's image calculation."""
    image = star.compute_image(dim, 0.1*u.mas, wavelength)

    star_dir = Path("fluxes/star")
    if not star_dir.exists():
        star_dir.mkdir()

    centre = dim//2

    plt.imshow(image.value[0])
    plt.xlim(centre-20, centre+20)
    plt.ylim(centre-20, centre+20)
    plt.savefig(star_dir / f"dim{dim}_wl{wl.value}_star_image.pdf")
    plt.close()

    assert len(image[image != 0]) == 4
    assert image.shape == (1, dim, dim)
    assert image.unit == u.Jy
    assert np.max(image.value) < 0.1


@pytest.mark.parametrize("grid_type", ["linear", "logarithmic"])
def test_temp_gradient_compute_grid(
        temp_gradient: TempGradient, grid_type: str) -> None:
    """Tests the hankel component's grid calculation."""
    OPTIONS.model.gridtype = grid_type
    radius = temp_gradient.compute_internal_grid(512)
    assert radius.unit == u.mas
    assert radius.shape == (512, )
    assert radius[0].value == temp_gradient.params["rin"].value\
        and radius[-1].value == temp_gradient.params["rout"].value

    OPTIONS.model.gridtype = "logarithmic"


def test_temp_gradient_compute_brightness():
    ...


def test_temp_gradient_flux(
        temp_gradient: TempGradient, wavelength: u.um) -> None:
    """Tests the calculation of the total flux."""
    flux = temp_gradient.compute_flux(wavelength)
    assert flux.shape == (wavelength.size, 1)


# TODO: Write test for hankel transform itself and compare it to ring model (aspro).
# and skewed ring model of aspro
# TODO: Write here check if higher orders are implemented
@pytest.mark.parametrize("order", [0, 1, 2, 3])
def test_temp_gradient_hankel_transform(
        temp_gradient: TempGradient,
        order: int, wavelength: u.um) -> None:
    """Tests the hankel component's hankel transformation."""
    radius = temp_gradient.compute_internal_grid(512)

    OPTIONS.model.modulation = order

    vis, vis_mod = temp_gradient.compute_hankel_transform(
            radius, READOUT.vis2.ucoord, READOUT.vis2.vcoord, wavelength)

    assert vis.shape == (wavelength.size, 6)
    assert vis_mod.shape == (wavelength.size, 6, order)
    assert vis.unit == u.Jy and vis_mod.unit == u.Jy

    OPTIONS.model.modulation = 0

# TODO: Add tests for the wavelength
@pytest.mark.parametrize("order", [0, 1, 2, 3])
def test_temp_gradient_vis_func(
        temp_gradient: TempGradient,
        order: int, wavelength: u.um) -> None:
    """Tests the hankel component's hankel transformation."""
    OPTIONS.model.modulation = order

    vis = temp_gradient.vis_func(
            READOUT.vis2.ucoord, READOUT.vis2.vcoord,
            wavelength)
    assert vis.shape == (wavelength.size, 6)
    assert isinstance(vis, np.ndarray)

    t3 = temp_gradient.vis_func(
            READOUT.t3.u123coord, READOUT.t3.v123coord, wavelength)

    assert t3.shape == (wavelength.size, 3, 4)
    assert isinstance(vis, np.ndarray)

    OPTIONS.model.modulation = 0


# TODO: Extend this test to account for multiple files (make files an input)
@pytest.mark.parametrize(
        "dim", [4096, 2096, 1024, 512, 256, 128, 64, 32])
def test_temp_gradient_resolution(dim: int, wavelength: u.um) -> None:
    """Tests the hankel component's resolution."""
    temp_gradient = TempGradient(
            rin=0.5, rout=3, pa=33,
            elong=0.5, dim=dim, a=0.5, phi=33,
            inner_temp=1500, q=0.5)
    temp_gradient.optically_thick = True
    temp_gradient.asymmetric = True

    OPTIONS.model.modulation = 1
    start_time_vis = time.perf_counter()
    _ = temp_gradient.vis_func(
            READOUT.vis2.ucoord, READOUT.vis2.vcoord, wavelength)
    end_time_vis = time.perf_counter()-start_time_vis

    start_time_cphase = time.perf_counter()
    _ = temp_gradient.vis_func(
            READOUT.t3.u123coord, READOUT.t3.v123coord, wavelength)
    end_time_cphase = time.perf_counter()-start_time_cphase

    vis_data = {"Dimension (px)": [dim],
                "Computation Time (s)": [end_time_vis]}

    t3_data = {"Dimension (px)": [dim],
               "Computation Time (s)": [end_time_cphase]}

    if CALCULATION_FILE.exists():
        df = pd.read_excel(CALCULATION_FILE, sheet_name="Vis")
        new_df = pd.DataFrame(vis_data)
        df = pd.concat([df, new_df])
    else:
        df = pd.DataFrame(vis_data)

    with pd.ExcelWriter(CALCULATION_FILE, engine="openpyxl",
                        mode="a", if_sheet_exists="replace") as writer:
        df.to_excel(writer, sheet_name="Vis", index=False)

    if CALCULATION_FILE.exists():
        df = pd.read_excel(CALCULATION_FILE, sheet_name="T3")
        new_df = pd.DataFrame(t3_data)
        df = pd.concat([df, new_df])
    else:
        df = pd.DataFrame(t3_data)

    with pd.ExcelWriter(CALCULATION_FILE, engine="openpyxl",
                        mode="a", if_sheet_exists="replace") as writer:
        df.to_excel(writer, sheet_name="T3", index=False)

    OPTIONS.model.modulation = 0


def test_assemble_components() -> None:
    """Tests the model's assemble_model method."""
    param_names = ["rin", "p", "a", "phi",
                   "cont_weight", "pa", "elong"]
    values = [1.5, 0.5, 0.3, 33, 0.2, 45, 1.6]
    limits = [[0, 20], [0, 1], [0, 1],
              [0, 360], [0, 1], [0, 360], [1, 50]]
    params = {name: Parameter(**STANDARD_PARAMETERS[name])
              for name in param_names}
    for value, limit, param in zip(values, limits, params.values()):
        param.set(*limit)
        param.value = value
    shared_params = {"p": params["p"]}
    del params["p"]

    components_and_params = [["Star", params],
                             ["GreyBody", params]]
    components = assemble_components(components_and_params, shared_params)
    assert isinstance(components[0], Star)
    assert isinstance(components[1], GreyBody)
    assert all(param not in components[0].params
               for param in param_names if param not in ["pa", "elong"])
    assert all(param in components[1].params for param in param_names)
    assert all(components[1].params[name].value == value
               for name, value in zip(["pa", "elong"], values[-2:]))
    assert all(components[1].params[name].value == value
               for name, value in zip(param_names, values))
    assert all([components[0].params[name].min,
                components[0].params[name].max] == limit
               for name, limit in zip(["pa", "elong"], limits[-2:]))
    assert all([components[1].params[name].min,
                components[1].params[name].max] == limit
               for name, limit in zip(param_names, limits))
