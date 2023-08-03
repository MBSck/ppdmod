from pathlib import Path
from typing import Dict, Tuple

import astropy.units as u
import numpy as np
import pytest

from ppdmod.component import NumericalComponent
from ppdmod.custom_components import Star, TemperatureGradient,\
    AsymmetricSDTemperatureGradient, AsymmetricSDGreyBody,\
    AsymmetricSDGreyBodyContinuum
from ppdmod.parameter import Parameter
from ppdmod.readout import ReadoutFits
from ppdmod.options import OPTIONS
from ppdmod.utils import opacity_to_matisse_opacity, linearly_combine_opacities,\
    calculate_intensity, get_binned_dimension


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
    return {"dim": 512, "dist": 145, "eff_temp": 7800, "eff_radius": 1.8}


@pytest.fixture
def temp_gradient_parameters() -> Dict[str, float]:
    """The temperature gradient's parameters."""
    return {"rin": 0.5, "rout": 100, "dust_mass": 0.11, "q": 0.5,
            "inner_temp": 1500, "pixel_size": 0.1, "p": 0.5}


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
                    wavelength: u.um,
                    threshold: float) -> None:
    """Tests the star's image calculation."""
    image = star._image_function(*grid, wavelength)
    assert image.unit == u.Jy
    assert np.max(image.value) < threshold


def test_star_visibility_function(star: Star,
                                  wavelength: u.um) -> None:
    """Tests the star's complex visibility function calculation."""
    intensity = calculate_intensity(star.params["eff_temp"](),
                                    wavelength,
                                    star.stellar_radius_angular).value
    complex_visibility = star._visibility_function(wavelength)
    assert complex_visibility.shape == (star.params["dim"](),
                                        star.params["dim"]())
    assert np.all(complex_visibility == intensity)


def test_temperature_gradient_init(temp_gradient: TemperatureGradient) -> None:
    """Tests the asymmetric's initialization."""
    assert "pa" in temp_gradient.params
    assert "elong" in temp_gradient.params
    assert "dist" in temp_gradient.params
    assert "inner_temp" in temp_gradient.params
    assert "eff_temp" in temp_gradient.params
    assert "eff_radius" in temp_gradient.params
    assert "q" in temp_gradient.params
    assert "p" in temp_gradient.params
    assert "dust_mass" in temp_gradient.params
    assert "kappa_abs" in temp_gradient.params


def test_asymmetric_temperature_gradient_init(
        asym_temp_gradient: AsymmetricSDTemperatureGradient) -> None:
    """Tests the asymmetric temperature gradient's initialization."""
    assert "a" in asym_temp_gradient.params
    assert "phi" in asym_temp_gradient.params


def test_asymmetric_temperature_gradient_init(
        asym_continuum_grey_body: AsymmetricSDGreyBodyContinuum) -> None:
    """Tests the asymmetric temperature gradient's initialization."""
    assert "kappa_cont" in asym_continuum_grey_body.params
    assert "cont_weight" in asym_continuum_grey_body.params


def test_temperature_gradient_surface_density_profile(
        temp_gradient: TemperatureGradient,
        grid: Tuple[u.mas, u.mas],
        radius: u.mas) -> None:
    """Tests the temperature gradient's surface density profile calculation."""
    surface_density = temp_gradient._calculate_surface_density_profile(radius, *grid)
    assert surface_density.unit == u.g/u.cm**2


def test_asym_temperature_gradient_azimuthal_modulation(
        asym_temp_gradient: AsymmetricSDTemperatureGradient,
        grid: Tuple[u.mas, u.mas]) -> None:
    """Tests the temperature gradient's azimuthal modulation."""
    asym_temp_gradient.params["a"].value = 0.5
    asym_temp_gradient.params["phi"].value = 33
    assert asym_temp_gradient._calculate_azimuthal_modulation(*grid).unit == u.one


def test_temperature_gradient_optical_depth(
        temp_gradient: TemperatureGradient,
        grid: Tuple[u.mas, u.mas],
        radius: u.mas,
        wavelength: u.um,
        opacity: Parameter) -> None:
    """Tests the temperature gradient's optical depth calculation."""
    temp_gradient.params["kappa_abs"] = opacity
    optical_depth =temp_gradient._calculate_optical_depth(
        radius, *grid, wavelength)
    assert optical_depth.unit == u.one


def test_temperature_gradient_temperature_profile(
        temp_gradient: TemperatureGradient,
        radius: u.mas) -> None:
    """Tests the temperature gradient's temperature profile"""
    temp_profile = temp_gradient._calculate_temperature_profile(radius)
    assert temp_profile.unit == u.K


def test_asym_grey_body_optical_depth(
        asym_continuum_grey_body: AsymmetricSDGreyBodyContinuum,
        grid: Tuple[u.mas, u.mas],
        radius: u.mas,
        wavelength: u.um,
        opacity: Parameter,
        continuum_opacity: Parameter) -> None:
    """Tests the asymmetric gradient's optical depth calculation."""
    asym_continuum_grey_body.params["kappa_cont"] = continuum_opacity
    asym_continuum_grey_body.params["cont_weight"].value = 0.5
    optical_depth = asym_continuum_grey_body._calculate_optical_depth(
        radius, *grid, wavelength)
    assert optical_depth.unit == u.one


def test_asym_grey_body_temperature_profile(
        asym_grey_body: AsymmetricSDGreyBody,
        radius: u.mas) -> None:
    """Tests the temperature gradient's temperature profile."""
    temp_profile = asym_grey_body._calculate_temperature_profile(radius)
    assert temp_profile.shape == radius.shape
    assert temp_profile.unit == u.K


def test_temperature_gradient_image_function(
        temp_gradient: TemperatureGradient,
        grid: Tuple[u.mas, u.mas],
        wavelength: u.um,
        opacity: Parameter) -> None:
    """Tests the temperature gradient's image function."""
    OPTIONS["fourier.binning"] = None
    temp_gradient.params["kappa_abs"] = opacity
    image = temp_gradient._image_function(*grid, wavelength)
    assert image.shape == grid[0].shape
    assert image.unit == u.Jy

    OPTIONS["fourier.binning"] = 1
    image = temp_gradient._image_function(*grid, wavelength)
    assert image.shape == tuple(np.array(grid[0].shape)//2)
    assert image.unit == u.Jy


def test_numerical_component_calculate_complex_visibility(
        temp_gradient: TemperatureGradient,
        wavelength: u.um,
        opacity: Parameter) -> None:
    """Tests the numerical component's complex visibility
    function calculation."""
    OPTIONS["fourier.binning"] = None
    temp_gradient.params["kappa_abs"] = opacity
    dim = temp_gradient.params["dim"]()
    complex_visibility = temp_gradient.calculate_complex_visibility(wavelength)
    assert np.all(complex_visibility != 0)
    assert complex_visibility.shape == (dim, dim)

    OPTIONS["fourier.binning"] = 2
    binned_dim = get_binned_dimension(dim, OPTIONS["fourier.binning"])
    complex_visibility = temp_gradient.calculate_complex_visibility(wavelength)
    assert np.all(complex_visibility != 0)
    assert complex_visibility.shape == (binned_dim, binned_dim)
