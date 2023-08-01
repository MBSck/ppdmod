from pathlib import Path

import astropy.units as u
import numpy as np
import pytest

from ppdmod.readout import ReadoutFits
from ppdmod.utils import calculate_intensity, qval_to_opacity,\
    opacity_to_matisse_opacity, linearly_combine_opacities


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


def test_exection_time():
    ...


def test_calculate_intensity(wavelength: u.um) -> None:
    """Tests the intensity calculation [Jy/px]."""
    intensity = calculate_intensity(7800*u.K,
                                    wavelength,
                                    (0.1*u.mas))
    assert intensity.unit == u.Jy
    assert intensity.value < 0.1


def test_pad_image():
    ...


def test_get_binned_dimension():
    ...


def test_rebin_image():
    ...


def test_get_next_power_of_two():
    ...


def test_angular_to_distance():
    ...


def test_qval_to_opacity(qval_file_dir: Path) -> np.ndarray:
    """Tests the readout of a qval file."""
    qval_file = qval_file_dir / "Q_Am_Mgolivine_Jae_DHS_f1.0_rv0.1.dat"
    wavelength, opacity = qval_to_opacity(qval_file)
    assert wavelength.unit == u.um
    assert opacity.unit == u.cm**2/u.g


# NOTE: This test, tests nothing.
def test_opacity_to_matisse_opacity(
        qval_file_dir: Path, wavelength_solution: u.um) -> None:
    """Tests the interpolation to the MATISSE wavelength grid."""
    qval_file = qval_file_dir / "Q_SILICA_RV0.1.DAT"
    continuum_opacity = opacity_to_matisse_opacity(wavelength_solution,
                                                   qval_file=qval_file)
    assert continuum_opacity.unit == u.cm**2/u.g


def test_linearly_combine_opacities(
        qval_file_dir: Path, wavelength_solution: u.um) -> None:
    """Tests the linear combination of interpolated wavelength grids."""
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
    assert opacity.unit == u.cm**2/u.g
