from pathlib import Path
from typing import Union, Any

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ppdmod import utils
from ppdmod.readout import ReadoutFits
from ppdmod.options import OPTIONS


@pytest.fixture
def qval_file_dir() -> Path:
    """The qval-file directory."""
    return Path("/Users/scheuck/Data/opacities/QVAL")


@pytest.fixture
def fits_file() -> Path:
    """A MATISSE (.fits)-file."""
    path = Path("/Users/scheuck/Data/reduced_data/hd142666/matisse")
    return path /\
        "hd_142666_2022-04-23T03_05_25:2022-04-23T02_28_06_AQUARIUS_FINAL_TARGET_INT.fits"


@pytest.fixture
def wavelength() -> u.m:
    """A wavelength grid."""
    return (8.28835527e-06*u.m).to(u.um)


@pytest.fixture
def wavelength_solution(fits_file: Path) -> u.um:
    """The wavelength solution of a MATISSE (.fits)-file."""
    return (ReadoutFits(fits_file).wavelength*u.m).to(u.um)


def test_exection_time() -> None:
    ...


def test_set_list_from_args() -> None:
    """Tests the set lists from args function."""
    args = [1, 2, 3]
    args = utils.set_tuple_from_args(*args)
    assert args == (1, 2, 3)

    args = 1, 2, 3
    args = utils.set_tuple_from_args(*args)
    assert args == (1, 2, 3)


def test_set_fit_wavelenghts() -> None:
    """Tests the set fit wavelenghts function."""
    utils.set_fit_wavelengths([4.78301581e-06, 8.28835527e-06])
    assert np.array_equal(OPTIONS["fit.wavelengths"],
                          ([4.78301581e-06, 8.28835527e-06]*u.m).to(u.um))

    utils.set_fit_wavelengths()
    assert not OPTIONS["fit.wavelengths"]


def test_set_data(fits_file: Path) -> None:
    """Tests the automatic data procurrment from one
    or multiple (.fits)-files."""
    utils.set_fit_wavelengths([4.78301581e-06, 8.28835527e-06])
    breakpoint()
    utils.set_data(fits_file)


def test_uniform_disk() -> None:
    """Tests the calculation of a uniform disk's brightness."""
    uniform_disk = utils.uniform_disk(0.1*u.mas, 512,
                                      diameter=4*u.mas)
    assert uniform_disk.shape == (512, 512)
    assert uniform_disk.unit == u.one


def test_uniform_disk_vis(wavelength: u.um) -> None:
    """Tests the calculation of a uniform disk's complex
    visibility function."""
    ucoord = np.linspace(80, 100, 20)*u.m
    uniform_disk_vis = utils.uniform_disk_vis(4*u.mas,
                                              ucoord, ucoord*0,
                                              wavelength)
    assert uniform_disk_vis.unit == u.one
    assert np.array_equal(np.real(uniform_disk_vis), uniform_disk_vis)


@pytest.mark.parametrize(
    "number, expected", [(22.5, 32), (100, 128),
                         (560, 1024), (100/0.1, 1024),
                         (200/0.1, 2048)])
def test_get_next_power_of_two(number: Union[int, float],
                               expected: int) -> None:
    """Tests the function that gets the next power of two."""
    assert utils.get_next_power_of_two(number) == expected


@pytest.mark.parametrize(
    "dimension, binning_factor, expected", [(1024, 2, 256), (1024, 1, 512),
                                            (1024, 3, 128), (2048, 1, 1024)])
def test_get_binned_dimension(dimension: int,
                              binning_factor: int, expected: int) -> None:
    """Tests if the binned dimension is properly calculated from
    the binning factor."""
    assert utils.get_binned_dimension(dimension, binning_factor) == expected


@pytest.mark.parametrize(
    "binning_factor, expected", [(1, 256), (2, 128), (3, 64)])
def test_rebin_image(binning_factor: int, expected: int) -> None:
    """Tests the rebinning of an image"""
    image = utils.uniform_disk(0.1*u.mas, 512, diameter=4*u.mas)
    rebinned_image = utils.rebin_image(image, binning_factor)

    binning_dir = Path("binning")
    if not binning_dir.exists():
        binning_dir.mkdir()
    _, (ax, bx) = plt.subplots(1, 2)

    ax.imshow(image.value)
    ax.set_title("Pre-binning")
    ax.set_xlabel("dim [px]")
    bx.imshow(rebinned_image.value)
    bx.set_title("After-binning")
    bx.set_xlabel("dim [px]")
    plt.savefig(binning_dir / f"Binning_factor_{binning_factor}.pdf",
                format="pdf")

    assert image.shape == (512, 512)
    assert rebinned_image.shape == (expected, expected)


@pytest.mark.parametrize(
    "padding_factor, expected", [(1, 128), (2, 256), (3, 512)])
def test_pad_image(padding_factor: int, expected: int) -> None:
    """Tests the padding of an image"""
    image = utils.uniform_disk(0.1*u.mas, 64)
    padded_image = utils.pad_image(image, padding_factor)

    padding_dir = Path("padding")
    if not padding_dir.exists():
        padding_dir.mkdir()
    _, (ax, bx) = plt.subplots(1, 2)

    ax.imshow(image.value, vmin=0.0, vmax=0.1)
    ax.set_title("Pre-padding")
    ax.set_xlabel("dim [px]")
    bx.imshow(padded_image.value)
    bx.set_title("After-padding")
    bx.set_xlabel("dim [px]")
    plt.savefig(padding_dir / f"Binning_factor_{padding_factor}.pdf",
                format="pdf")

    assert image.shape == (64, 64)
    assert padded_image.shape == (expected, expected)


def test_qval_to_opacity(qval_file_dir: Path) -> None:
    """Tests the readout of a qval file."""
    qval_file = qval_file_dir / "Q_Am_Mgolivine_Jae_DHS_f1.0_rv0.1.dat"
    wavelength, opacity = utils.qval_to_opacity(qval_file)
    assert wavelength.unit == u.um
    assert opacity.unit == u.cm**2/u.g


def test_transform_opacity():
    """Tests the opacity interpolation from one wavelength
    axis to another."""
    ...


# NOTE: This test, tests nothing.
def test_opacity_to_matisse_opacity(
        qval_file_dir: Path, wavelength_solution: u.um) -> None:
    """Tests the interpolation to the MATISSE wavelength grid."""
    qval_file = qval_file_dir / "Q_SILICA_RV0.1.DAT"
    continuum_opacity = utils.opacity_to_matisse_opacity(wavelength_solution,
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
    opacity = utils.linearly_combine_opacities(weights,
                                               qval_paths,
                                               wavelength_solution)
    assert opacity.unit == u.cm**2/u.g


@pytest.mark.parametrize(
    "distance, expected", [(2.06*u.km, 1*u.cm), (1*u.au, 725.27*u.km),
                           (1*u.lyr, 45_866_916*u.km), (1*u.pc, 1*u.au)])
def test_angular_to_distance(distance: u.Quantity,
                             expected: u.Quantity) -> None:
    """Tests the angular diameter equation to
    convert angules to distance.

    Test values for 1" angular diameter from wikipedia:
    https://en.wikipedia.org/wiki/Angular_diameter.
    """
    diameter = utils.angular_to_distance(1*u.arcsec, distance)
    assert np.isclose(diameter, expected.to(u.m), atol=1e-3)


def test_calculate_intensity(wavelength: u.um) -> None:
    """Tests the intensity calculation [Jy/px]."""
    intensity = utils.calculate_intensity(7800*u.K,
                                          wavelength,
                                          0.1*u.mas)
    assert intensity.unit == u.Jy
    assert intensity.value < 0.1
