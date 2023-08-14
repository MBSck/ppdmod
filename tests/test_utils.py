from pathlib import Path
from typing import Union, List

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ppdmod import utils
from ppdmod.custom_components import AsymmetricSDGreyBodyContinuum

from ppdmod.data import ReadoutFits


@pytest.fixture
def qval_file_dir() -> Path:
    """The qval-file directory."""
    return Path("data/qval")


@pytest.fixture
def qval_files(qval_file_dir: Path) -> List[Path]:
    files = ["Q_Am_Mgolivine_Jae_DHS_f1.0_rv0.1.dat",
             "Q_Am_Mgolivine_Jae_DHS_f1.0_rv1.5.dat",
             "Q_Am_Mgpyroxene_Dor_DHS_f1.0_rv1.5.dat",
             "Q_Fo_Suto_DHS_f1.0_rv0.1.dat",
             "Q_Fo_Suto_DHS_f1.0_rv1.5.dat",
             "Q_En_Jaeger_DHS_f1.0_rv1.5.dat"]
    return list(map(lambda x: qval_file_dir / x, files))


@pytest.fixture
def fits_file() -> Path:
    """A MATISSE (.fits)-file."""
    return Path("data/fits") /\
        "hd_142666_2022-04-23T03_05_25:2022-04-23T02_28_06_AQUARIUS_FINAL_TARGET_INT.fits"


@pytest.fixture
def wavelength() -> u.m:
    """A wavelength grid."""
    return (8.28835527e-06*u.m).to(u.um)


@pytest.fixture
def wavelengths() -> u.um:
    """A wavelength grid."""
    return ([8.28835527e-06, 1.02322101e-05]*u.m).to(u.um)


@pytest.fixture
def wavelength_solution(fits_file: Path) -> u.um:
    """The wavelength solution of a MATISSE (.fits)-file."""
    return ReadoutFits(fits_file).wavelength


@pytest.fixture
def rin() -> float:
    """The inner radius."""
    return 35.


@pytest.fixture
def temp_gradient_image(rin: float, wavelength: u.um) -> None:
    """A temperature gradient."""
    dim = utils.get_next_power_of_two(200/0.1)
    asym_grey_body = AsymmetricSDGreyBodyContinuum(
        dist=145, eff_temp=7800, eff_radius=1.8,
        dim=dim, rin=rin, a=0.3, phi=33,
        pixel_size=0.1, pa=45,
        elong=1.6, inner_sigma=2000, kappa_abs=1000,
        kappa_cont=3000, cont_weight=0.5, p=0.5)
    asym_grey_body.optically_thick = True
    return asym_grey_body.calculate_image(wavelength=wavelength)


def test_exection_time() -> None:
    ...


def test_make_workbook() -> None:
    ...


def test_set_list_from_args(wavelengths: u.um) -> None:
    """Tests the set lists from args function."""
    arguments = [1, 2, 3]
    arguments = utils.set_tuple_from_args(*arguments)
    assert arguments == (1, 2, 3)

    arguments = 1, 2, 3
    arguments = utils.set_tuple_from_args(*arguments)
    assert arguments == (1, 2, 3)

    arguments = [1, 2, 3]*u.m
    arguments = utils.set_tuple_from_args(*arguments)
    assert arguments == (1*u.m, 2*u.m, 3*u.m)

    arguments = utils.set_tuple_from_args(*wavelengths)
    assert arguments == ((8.28835527e-06*u.m).to(u.um),
                         (1.02322101e-05*u.m).to(u.um),)


def test_get_closest_indices(wavelength: u.um,
                             wavelengths: u.um,
                             wavelength_solution: u.um) -> None:
    """Tests the get_closest_indices function."""
    index = utils.get_closest_indices(wavelength, array=wavelength_solution)
    indices = utils.get_closest_indices(wavelengths, array=wavelength_solution)
    assert index.size > 0 and indices.size > 0


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


def test_binary() -> None:
    """Tests the calculation of a binary's brightness."""
    flux1, flux2 = 5*u.Jy, 2*u.Jy
    position1, position2 = [5, 10]*u.mas, [-10, -10]*u.mas
    binary = utils.binary(512, 0.1*u.mas,
                          flux1, flux2,
                          position1, position2)
    assert binary[binary != 0].size == 2
    assert binary.shape == (512, 512)
    assert binary.unit == u.Jy


def test_binary_vis(wavelength: u.um) -> None:
    """Tests the calculation of the binary's complex
    visibility function."""
    ucoord = np.linspace(80, 100, 20)*u.m
    flux1, flux2 = 5*u.Jy, 2*u.Jy
    position1, position2 = [5, 10]*u.mas, [-10, -10]*u.mas
    binary_vis = utils.binary_vis(flux1, flux2,
                                  ucoord, ucoord*0,
                                  position1, position2,
                                  wavelength)
    assert isinstance(binary_vis, np.ndarray)
    assert not np.array_equal(np.real(binary_vis), binary_vis)


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
    "binning_factor, expected", [(1, 1024), (2, 512), (3, 256),
                                 (4, 128), (5, 64), (6, 32)])
def test_rebin_image(rin: float, temp_gradient_image: u.mas,
                     binning_factor: int, expected: int) -> None:
    """Tests the rebinning of an image"""
    px, dim, dia = 0.1, 2048, 60
    ud_image = utils.uniform_disk(px*u.mas, dim, diameter=dia*u.mas)
    rebinned_ud = utils.rebin_image(ud_image, binning_factor)
    rebinned_temp_grad = utils.rebin_image(temp_gradient_image,
                                           binning_factor)

    binning_dir = Path("binning")
    if not binning_dir.exists():
        binning_dir.mkdir()

    _, (ax, bx) = plt.subplots(1, 2)
    ax.imshow(ud_image.value)
    ax.set_title("Pre-binning")
    ax.set_xlabel("dim [px]")
    bx.imshow(rebinned_ud.value)
    bx.set_title("After-binning")
    bx.set_xlabel("dim [px]")
    plt.savefig(binning_dir /
                f"ud_px{px}_dim{dim}_dia{dia}_bin{binning_factor}.pdf",
                format="pdf")
    plt.close()

    assert ud_image.shape == (dim, dim)
    assert rebinned_ud.shape == (expected, expected)

    _, (ax, bx) = plt.subplots(1, 2)
    ax.imshow(temp_gradient_image.value)
    ax.set_title("Pre-binning")
    ax.set_xlabel("dim [px]")
    bx.imshow(rebinned_temp_grad.value)
    bx.set_title("After-binning")
    bx.set_xlabel("dim [px]")
    plt.savefig(binning_dir /
                f"temp_gradient_rin{rin}_"
                f"px{px}_dim{dim}_dia{dia}_bin{binning_factor}.pdf",
                format="pdf")
    plt.close()

    assert temp_gradient_image.shape == (dim, dim)
    assert rebinned_temp_grad.shape == (expected, expected)


# NOTE: Padding high dimensions takes ages.
@pytest.mark.parametrize(
    "padding_factor, expected", [(1, 4096), (2, 8192)])
def test_pad_image(rin: float, temp_gradient_image: u.mas,
                   padding_factor: int, expected: int) -> None:
    """Tests the padding of an image"""
    dim, px = 2048, 0.1
    ud_image = utils.uniform_disk(px*u.mas, dim)
    padded_image = utils.pad_image(ud_image, padding_factor)
    padded_temp_gradient = utils.pad_image(
        temp_gradient_image, padding_factor)

    padding_dir = Path("padding")
    if not padding_dir.exists():
        padding_dir.mkdir()

    _, (ax, bx) = plt.subplots(1, 2)
    ax.imshow(ud_image.value, vmin=0.0, vmax=0.1)
    ax.set_title("Pre-padding")
    ax.set_xlabel("dim [px]")
    bx.imshow(padded_image.value)
    bx.set_title("After-padding")
    bx.set_xlabel("dim [px]")
    plt.savefig(padding_dir / f"ud_dim{dim}_px{px}_pad{padding_factor}.pdf",
                format="pdf")
    plt.close()

    assert ud_image.shape == (dim, dim)
    assert padded_image.shape == (expected, expected)

    _, (ax, bx) = plt.subplots(1, 2)
    ax.imshow(temp_gradient_image.value)
    ax.set_title("Pre-padding")
    ax.set_xlabel("dim [px]")
    bx.imshow(padded_temp_gradient.value)
    bx.set_title("After-padding")
    bx.set_xlabel("dim [px]")
    plt.savefig(padding_dir /
                f"temp_gradient_rin{rin}_"
                f"dim{dim}_px{px}_pad{padding_factor}.pdf",
                format="pdf")
    plt.close()

    assert temp_gradient_image.shape == (dim, dim)
    assert padded_image.shape == (expected, expected)


def test_qval_to_opacity(qval_file_dir: Path) -> None:
    """Tests the readout of a qval file."""
    qval_file = qval_file_dir / "Q_Am_Mgolivine_Jae_DHS_f1.0_rv0.1.dat"
    wavelength, opacity = utils.qval_to_opacity(qval_file)
    assert wavelength.unit == u.um
    assert opacity.unit == u.cm**2/u.g


@pytest.mark.parametrize(
    "file", ["Q_Am_Mgolivine_Jae_DHS_f1.0_rv0.1.dat",
             "Q_Am_Mgolivine_Jae_DHS_f1.0_rv1.5.dat",
             "Q_Am_Mgpyroxene_Dor_DHS_f1.0_rv1.5.dat",
             "Q_Fo_Suto_DHS_f1.0_rv0.1.dat",
             "Q_Fo_Suto_DHS_f1.0_rv1.5.dat",
             "Q_En_Jaeger_DHS_f1.0_rv1.5.dat"])
def test_transform_opacity(qval_file_dir: Path, file: str,
                           wavelength_solution: u.um) -> None:
    """Tests the opacity interpolation from one wavelength
    axis to another."""
    opacity_dir = Path("opacities")
    if not opacity_dir.exists():
        opacity_dir.mkdir()

    qval_file = qval_file_dir / file
    wavelength_grid, opacity = utils.qval_to_opacity(qval_file)
    ind = np.where(np.logical_and(wavelength_solution.min() < wavelength_grid,
                                  wavelength_grid < wavelength_solution.max()))
    wavelength_grid, opacity = wavelength_grid[ind], opacity[ind]
    matisse_opacity = utils.transform_opacity(
        wavelength_grid, opacity, wavelength_solution)

    _, (ax, bx) = plt.subplots(1, 2, figsize=(12, 5))
    ax.plot(wavelength_grid.value, opacity.value)
    ax.set_xlabel(r"$\lambda [um]$")
    ax.set_ylabel(r"$\kappa [cm^2/g]$")
    ax.set_title("Before interpolation.")
    bx.plot(wavelength_solution.value, matisse_opacity.value)
    bx.set_xlabel(r"$\lambda [um]$")
    bx.set_ylabel(r"$\kappa [cm^2/g]$")
    bx.set_title("After interpolation.")
    plt.savefig(opacity_dir / f"{qval_file.stem}.pdf", format="pdf")
    plt.close()

    assert opacity.unit == u.cm**2/u.g
    assert opacity.shape == wavelength_grid.shape
    assert matisse_opacity.unit == u.cm**2/u.g
    assert matisse_opacity.shape == wavelength_solution.shape


# NOTE: This test, tests nothing.
def test_opacity_to_matisse_opacity(
        qval_file_dir: Path, wavelength_solution: u.um) -> None:
    """Tests the interpolation to the MATISSE wavelength grid."""
    qval_file = qval_file_dir / "Q_SILICA_RV0.1.DAT"
    continuum_opacity = utils.opacity_to_matisse_opacity(wavelength_solution,
                                                         qval_file=qval_file)
    assert continuum_opacity.unit == u.cm**2/u.g


# TODO: Check if it is combined properly.
def test_linearly_combine_opacities(
        qval_file_dir: Path,
        qval_files: List[Path],
        wavelength_solution: u.um) -> None:
    """Tests the linear combination of interpolated wavelength grids."""
    opacity_dir = Path("opacities")
    if not opacity_dir.exists():
        opacity_dir.mkdir()

    weights = np.array([42.8, 9.7, 43.5, 1.1, 2.3, 0.6])/100
    opacity = utils.linearly_combine_opacities(
        weights, qval_files, wavelength_solution)

    weights = np.append(weights, [0.668])
    files = qval_files.copy()
    files.append(qval_file_dir / "Q_SILICA_RV0.1.DAT")
    opacity_with_silica = utils.linearly_combine_opacities(
        weights, files, wavelength_solution)

    _, (ax, bx) = plt.subplots(1, 2, figsize=(12, 5))
    ax.plot(wavelength_solution.value, opacity.value)
    ax.set_title("Combined Opacities")
    ax.set_xlabel(r"$\lambda [um]$")
    ax.set_ylabel(r"$\kappa [cm^2/g]$")
    bx.plot(wavelength_solution.value, opacity_with_silica.value)
    bx.set_title("Combined Opacities with Silica")
    bx.set_xlabel(r"$\lambda [um]$")
    bx.set_ylabel(r"$\kappa [cm^2/g]$")
    plt.savefig(opacity_dir / "combined_opacities.pdf", format="pdf")
    plt.close()

    assert opacity.unit == u.cm**2/u.g
    assert opacity_with_silica.unit == u.cm**2/u.g


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


def test_calculate_effective_baselines() -> None:
    """Tests the calculation of the effective baselines."""
    ...
