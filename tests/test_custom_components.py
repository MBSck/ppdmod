from pathlib import Path
from typing import Dict, Tuple

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ppdmod.component import Component, NumericalComponent
from ppdmod.custom_components import Star, TemperatureGradient,\
    AsymmetricSDTemperatureGradient, AsymmetricSDGreyBody,\
    AsymmetricSDGreyBodyContinuum, assemble_components
from ppdmod.parameter import STANDARD_PARAMETERS, Parameter
from ppdmod.data import ReadoutFits
from ppdmod.options import OPTIONS
from ppdmod.utils import opacity_to_matisse_opacity,\
    linearly_combine_opacities, get_new_dimension,\
    make_workbook, get_next_power_of_two, uniform_disk


FLUX_DIR = Path("fluxes")
if not FLUX_DIR.exists():
    FLUX_DIR.mdkir()

FLUX_FILE = Path("flux.xlsx")
FLUX_SHEET = "Fluxes for 13 um"

make_workbook(
    FLUX_FILE,
    {
        FLUX_SHEET: ["FOV [mas]",
                     "Dimension [px]",
                     "Dimension (Nearest Power of 2) [px]",
                     "Flux [Jy]",
                     "Pixel Size [mas/px]",
                     "Inner Radius [mas]"]
    })


@pytest.fixture
def wavelength() -> u.m:
    """A wavelenght grid."""
    return (13.000458e-6*u.m).to(u.um)


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
def opacity(qval_file_dir: Path,
            wavelength_solution: u.um) -> None:
    """A parameter containing the opacity."""
    weights = np.array([42.8, 9.7, 43.5, 1.1, 2.3, 0.6])/100
    qval_files = ["Q_Am_Mgolivine_Jae_DHS_f1.0_rv0.1.dat",
                  "Q_Am_Mgolivine_Jae_DHS_f1.0_rv1.5.dat",
                  "Q_Am_Mgpyroxene_Dor_DHS_f1.0_rv1.5.dat",
                  "Q_Fo_Suto_DHS_f1.0_rv0.1.dat",
                  "Q_Fo_Suto_DHS_f1.0_rv1.5.dat",
                  "Q_En_Jaeger_DHS_f1.0_rv1.5.dat"]
    qval_paths = list(map(lambda x: qval_file_dir / x, qval_files))
    opacity = linearly_combine_opacities(weights,
                                         qval_paths, wavelength_solution)
    return Parameter(name="kappa_abs", value=opacity,
                     wavelength=wavelength_solution,
                     unit=u.cm**2/u.g, free=False,
                     description="Dust mass absorption coefficient")


@pytest.fixture
def continuum_opacity(qval_file_dir: Path,
                      wavelength_solution: u.um) -> None:
    """A parameter containing the continuum opacity."""
    qval_file = qval_file_dir / "Q_SILICA_RV0.1.DAT"
    continuum_opacity = opacity_to_matisse_opacity(wavelength_solution,
                                                   qval_file=qval_file)
    return Parameter(name="kappa_cont",
                     value=continuum_opacity,
                     wavelength=wavelength_solution,
                     unit=u.cm**2/u.g, free=False,
                     description="Continuum dust mass absorption coefficient")


@pytest.fixture
def grid() -> u.mas:
    """A spatial grid."""
    numerical_component = NumericalComponent(dim=512, pixel_size=0.1)
    return numerical_component._calculate_internal_grid()


@pytest.fixture
def radius(grid: Tuple[u.Quantity[u.mas], u.Quantity[u.mas]]) -> u.mas:
    """A radius based on a spatial grid"""
    return np.hypot(*grid)


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
def star(star_parameters: Dict[str, float]) -> Star:
    """Initializes a star component."""
    return Star(**star_parameters)


@pytest.fixture
def temp_gradient(star_parameters: Dict[str, float],
                  temp_gradient_parameters: Dict[str, float]
                  ) -> TemperatureGradient:
    """Initializes a temperature gradient component."""
    return TemperatureGradient(**star_parameters,
                               **temp_gradient_parameters)


@pytest.fixture
def asym_temp_gradient(star_parameters: Dict[str, float],
                       temp_gradient_parameters: Dict[str, float]
                       ) -> AsymmetricSDTemperatureGradient:
    """Initializes an asymmetric temperature gradient component."""
    return AsymmetricSDTemperatureGradient(**star_parameters,
                                           **temp_gradient_parameters)


@pytest.fixture
def asym_grey_body(star_parameters: Dict[str, float],
                   temp_gradient_parameters: Dict[str, float]
                   ) -> AsymmetricSDGreyBody:
    """Initializes an asymmetric temperature gradient component."""
    asym_grey_body = AsymmetricSDGreyBody(**star_parameters,
                                          **temp_gradient_parameters)
    asym_grey_body.params["q"] = 0
    return asym_grey_body


@pytest.fixture
def asym_continuum_grey_body(star_parameters: Dict[str, float],
                             temp_gradient_parameters: Dict[str, float]
                             ) -> AsymmetricSDGreyBodyContinuum:
    """Initializes an asymmetric temperature gradient component."""
    asym_grey_body = AsymmetricSDGreyBodyContinuum(**star_parameters,
                                                   **temp_gradient_parameters)
    asym_grey_body.params["q"] = 0
    return asym_grey_body


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


@pytest.mark.parametrize("wavelength, threshold",
                         [(8*u.um, 0.1), (9*u.um, 0.1),
                          (10*u.um, 0.1), (11*u.um, 0.1)])
def test_star_image(star: Star,
                    grid: Tuple[u.Quantity[u.mas], u.Quantity[u.mas]],
                    wavelength: u.um, threshold: float) -> None:
    """Tests the star's image calculation."""
    image = star._image_function(*grid, wavelength)
    assert image.unit == u.Jy
    assert np.max(image.value) < threshold


def test_star_visibility_function(star: Star,
                                  wavelength: u.um) -> None:
    """Tests the star's complex visibility function calculation."""
    complex_visibility = star._visibility_function(wavelength)
    assert complex_visibility.shape == (star.params["dim"](),
                                        star.params["dim"]())


def test_temperature_gradient_init(
        temp_gradient: TemperatureGradient) -> None:
    """Tests the asymmetric's initialization."""
    assert "pa" in temp_gradient.params
    assert "elong" in temp_gradient.params
    assert "dist" in temp_gradient.params
    assert "inner_temp" in temp_gradient.params
    assert "eff_temp" in temp_gradient.params
    assert "eff_radius" in temp_gradient.params
    assert "q" in temp_gradient.params
    assert "p" in temp_gradient.params
    assert "inner_sigma" in temp_gradient.params
    assert "kappa_abs" in temp_gradient.params
    assert "a" in temp_gradient.params
    assert "phi" in temp_gradient.params
    assert "kappa_cont" in temp_gradient.params
    assert "cont_weight" in temp_gradient.params


def test_temperature_gradient_image_function(
        temp_gradient: TemperatureGradient,
        grid: Tuple[u.Quantity[u.mas], u.Quantity[u.mas]],
        wavelength: u.um, opacity: Parameter) -> None:
    """Tests the temperature gradient's image function."""
    temp_gradient.params["kappa_abs"] = opacity
    image = temp_gradient._image_function(*grid, wavelength)
    assert image.shape == grid[0].shape
    assert image.unit == u.Jy

    OPTIONS["fourier.binning"] = 1
    image = temp_gradient._image_function(*grid, wavelength)
    OPTIONS["fourier.binning"] = None
    assert image.shape == tuple(np.array(grid[0].shape)//2)
    assert image.unit == u.Jy


def test_numerical_component_calculate_complex_visibility(
        temp_gradient: TemperatureGradient,
        wavelength: u.um, opacity: Parameter) -> None:
    """Tests the numerical component's complex visibility
    function calculation."""
    temp_gradient.params["kappa_abs"] = opacity
    dim = temp_gradient.params["dim"]()
    complex_visibility = temp_gradient.calculate_complex_visibility(wavelength)
    assert np.all(complex_visibility != 0)
    assert complex_visibility.shape == (dim, dim)

    OPTIONS["fourier.binning"] = 2
    binned_dim = get_new_dimension(dim, OPTIONS["fourier.binning"])
    complex_visibility = temp_gradient.calculate_complex_visibility(wavelength)
    OPTIONS["fourier.binning"] = None
    assert np.all(complex_visibility != 0)
    assert complex_visibility.shape == (binned_dim, binned_dim)

# TODO: Include more tests that also check fo different dimensions and such.
def test_calculate_image(wavelength: u.um) -> None:
    """Tests the image calculation for the normal and the matryoshka method."""
    rin, pixel_size, dim = 0.5*u.mas, 0.1*u.mas, 1024
    ud = uniform_disk(pixel_size, dim, diameter=4*u.mas)
    asym_grey_body = AsymmetricSDGreyBodyContinuum(
        dist=145, eff_temp=7800, eff_radius=1.8,
        dim=dim, rin=rin, a=0.3, phi=33,
        pixel_size=pixel_size.value, pa=45,
        elong=1.6, inner_sigma=1e-3, kappa_abs=1000,
        kappa_cont=1500, cont_weight=0.5, p=0.5)
    asym_grey_body.optically_thick = False

    OPTIONS["model.matryoshka"] = True
    OPTIONS["model.matryoshka.binning_factors"] = [2, 0, 1]

    image = asym_grey_body.calculate_image(
            dim, pixel_size, wavelength=wavelength)

    plt.imshow(image.value)
    plt.title("Temperature Gradient")
    plt.xlabel("dim [px]")
    plt.savefig(FLUX_DIR /
                f"matryoshka_method.pdf", format="pdf")
    plt.close()
    OPTIONS["model.matryoshka"] = False


# NOTE: Fov 420 is 8192 and 820 is 16xxx
# (not much change for the lower resolutions).
@pytest.mark.parametrize(
    "rin, fov, pixel_size", [(rin, fov, pixel_size)
                             for rin in [1, 2, 6, 10]*u.mas
                             for pixel_size in range(1, 3)*u.mas/10
                             for fov in [20, 40, 60, 120, 220]])
def test_flux_resolution(
        rin: u.mas, pixel_size: u.mas,
        fov: int, wavelength: u.um) -> None:
    """Tests the resolution of the flux for different pixel sizes
    and field of views."""
    dim = get_next_power_of_two(fov/pixel_size.value)
    asym_grey_body = AsymmetricSDGreyBodyContinuum(
        dist=145, eff_temp=7800, eff_radius=1.8,
        dim=dim, rin=rin, a=0.3, phi=33,
        pixel_size=pixel_size.value, pa=45,
        elong=1.6, inner_sigma=1e-3, kappa_abs=1000,
        kappa_cont=1500, cont_weight=0.5, p=0.5)
    asym_grey_body.optically_thick = False

    image = asym_grey_body.calculate_image(
            dim, pixel_size, wavelength=wavelength)
    flux = np.nansum(image)

    data = {"FOV [mas]": [fov],
            "Dimension [px]": [fov/pixel_size.value],
            "Dimension (Nearest Power of 2) [px]": [dim],
            "Flux [Jy]": [np.around(flux, 8).value],
            "Pixel Size [mas/px]": [pixel_size],
            "Inner Radius [mas]": [rin.value]}
    if FLUX_FILE.exists():
        df = pd.read_excel(FLUX_FILE, sheet_name=FLUX_SHEET)
        new_df = pd.DataFrame(data)
        df = pd.concat([df, new_df])
    else:
        df = pd.DataFrame(data)
    with pd.ExcelWriter(FLUX_FILE, engine="openpyxl",
                        mode="a", if_sheet_exists="replace") as writer:
        df.to_excel(writer, sheet_name=FLUX_SHEET, index=False)

    plt.imshow(image.value)
    plt.title("Temperature Gradient")
    plt.xlabel("dim [px]")
    plt.savefig(FLUX_DIR /
                f"temperature_gradient_dim{dim}_rin{rin.value}_px{pixel_size.value}.pdf",
                format="pdf")
    plt.close()


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
                             ["AsymmetricSDGreyBodyContinuum", params]]
    components = assemble_components(components_and_params, shared_params)
    assert isinstance(components[0], Star)
    assert isinstance(components[1], AsymmetricSDGreyBodyContinuum)
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
                             ["AsymmetricSDGreyBodyContinuum", params]]
    components = assemble_components(components_and_params, shared_params)
    assert isinstance(components[0], Star)
    assert isinstance(components[1], AsymmetricSDGreyBodyContinuum)
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


