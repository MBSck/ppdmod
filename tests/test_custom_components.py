from pathlib import Path
from typing import Dict, Tuple

import astropy.units as u
import numpy as np
import pytest

from ppdmod.component import NumericalComponent
from ppdmod.custom_components import Star, TemperatureGradient,\
    AsymmetricSDGreyBody, AsymmetricSDGreyBodyContinuum
from ppdmod.parameter import Parameter
from ppdmod.readout import ReadoutFits
from ppdmod.utils import opacity_to_matisse_opacity, linearly_combine_opacities


@pytest.fixture
def wavelength() -> u.m:
    """A wavelenght grid."""
    return (8.28835527e-06*u.m).to(u.um)


@pytest.fixture
def wavelength_solution() -> u.um:
    """A MATISSE (.fits)-file."""
    path = Path("/Users/scheuck/Data/reduced_data/hd142666/matisse")
    file = Path("hd_142666_2022-04-23T03_05_25:2022-04-23T02_28_06_AQUARIUS_FINAL_TARGET_INT.fits")
    return (ReadoutFits(path / file).wavelength*u.m).to(u.um)


@pytest.fixture
def qval_file_dir() -> Path:
    """The qval-file directory."""
    return Path("/Users/scheuck/Data/opacities/QVAL")


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
def radius(grid: Tuple[u.mas, u.mas]) -> u.mas:
    """A radius based on a spatial grid"""
    return np.hypot(*grid)


@pytest.fixture
def star_parameters() -> Dict[str, float]:
    """The star's parameters"""
    return {"dist": 145, "eff_temp": 7800, "eff_radius": 1.8}


@pytest.fixture
def star(star_parameters: Dict[str, float]) -> Star:
    """Initializes a star component."""
    return Star(**star_parameters)


@pytest.fixture
def temp_gradient(star_parameters: Dict[str, float]
                  ) -> TemperatureGradient:
    """Initializes a temperature gradient component."""
    return TemperatureGradient(**star_parameters)


def test_star_init(star: Star) -> None:
    """Tests the star's initialization."""
    assert "dist" in star.params
    assert "eff_temp" in star.params
    assert "eff_radius" in star.params


def test_star_stellar_radius_angular(star: Star) -> None:
    """Tests the stellar radius conversion to angular radius."""
    assert star.stellar_radius_angular.unit == u.rad


@pytest.mark.parametrize("wavelength, threshold",
                         [(8*u.um, 0.1), (9*u.um, 0.1),
                          (10*u.um, 0.1), (11*u.um, 0.1)])
def test_star_image(star: Star,
                    grid: Tuple[u.mas, u.mas],
                    wavelength: u.m,
                    threshold: float) -> None:
    """Tests the star's image calculation."""
    image = star._image_function(*grid, wavelength)
    assert image.unit == u.Jy
    assert np.max(image.value) < threshold


def test_temperature_gradient_init(temp_gradient: TemperatureGradient) -> None:
    """Tests the TemperatureGradient's initialization."""
    assert "dist" in temp_gradient.params
    assert "inner_temp" in temp_gradient.params
    assert "eff_temp" in temp_gradient.params
    assert "eff_radius" in temp_gradient.params
    assert "a" in temp_gradient.params
    assert "phi" in temp_gradient.params
    assert "q" in temp_gradient.params
    assert "p" in temp_gradient.params
    assert "Mdust" in temp_gradient.params
    assert "kappa_abs" in temp_gradient.params
    # assert "kappa_cont" in temp_gradient.params
    # assert "cont_weight" in temp_gradient.params


def test_temperature_gradient_azimuthal_modulation(
        temp_gradient: TemperatureGradient,
        grid: Tuple[u.mas, u.mas]) -> None:
    """Tests the temperature gradient's azimuthal modulation."""
    temp_gradient.params["a"].value = 0.5
    temp_gradient.params["phi"].value = 33
    assert temp_gradient._calculate_azimuthal_modulation(*grid).unit == u.one


def test_temperature_gradient_surface_density_profile(
        temp_gradient: TemperatureGradient,
        grid: Tuple[u.mas, u.mas],
        radius: u.mas) -> None:
    """Tests the temperature gradient's surface density profile calculation."""
    temp_gradient.params["rin"].value = 0.5
    temp_gradient.params["rout"].value = 100
    temp_gradient.params["Mdust"].value = 0.11
    surface_density = temp_gradient._calculate_surface_density_profile(radius, *grid)
    assert surface_density.unit == u.g/u.cm**2


# def test_temperature_gradient_optical_depth(
#         temp_gradient: TemperatureGradient,
#         grid: Tuple[u.mas, u.mas],
#         radius: u.mas,
#         wavelength: u.um,
#         opacity: u.cm**2/u.g,
#         continuum_opacity: u.cm**2/u.g) -> None:
#     """Tests the temperature gradient's optical depth calculation."""
#     breakpoint()
#     optical_depth = temp_gradient._calculate_surface_density_profile(
#         radius, *grid, wavelenght)
