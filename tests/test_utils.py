from pathlib import Path
from typing import Union, List

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ppdmod import utils
from ppdmod.custom_components import AsymmetricSDGreyBodyContinuum
from ppdmod.options import OPTIONS

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
def fits_files() -> Path:
    """A MATISSE (.fits)-file."""
    fits_files = [
        "hd_142666_2022-04-23T03_05_25:2022-04-23T02_28_06_HAWAII-2RG_FINAL_TARGET_INT.fits",
        "hd_142666_2022-04-23T03_05_25:2022-04-23T02_28_06_AQUARIUS_FINAL_TARGET_INT.fits"]
    return [Path("data/fits") / file for file in fits_files]


@pytest.fixture
def wavelength() -> u.m:
    """A wavelength grid."""
    return (8.28835527e-06*u.m).to(u.um)


@pytest.fixture
def wavelengths() -> u.um:
    """A wavelength grid."""
    return [3.520375, 8.502626, 10.001093]*u.um


@pytest.fixture
def wavelength_solution_l_band(fits_files: Path) -> u.um:
    """The wavelength solution of a MATISSE (.fits)-file."""
    return ReadoutFits(fits_files[0]).wavelength


@pytest.fixture
def wavelength_solution(fits_file: Path) -> u.um:
    """The wavelength solution of a MATISSE (.fits)-file."""
    return ReadoutFits(fits_file).wavelength


@pytest.fixture
def high_wavelength_solution() -> u.um:
    """The wavelength solution of a MATISSE (.fits)-file."""
    return np.load(Path("data") / "high_wavelength_solution.npy")*u.um


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


def test_get_closest_indices(
        wavelength: u.um, wavelengths: u.um,
        wavelength_solution_l_band: u.um,
        wavelength_solution: u.um) -> None:
    """Tests the get_closest_indices function."""
    index = utils.get_closest_indices(wavelength, array=wavelength_solution)
    indices = utils.get_closest_indices(wavelengths, array=wavelength_solution)
    indices_l_band = utils.get_closest_indices(
        wavelengths, array=wavelength_solution_l_band)
    assert len(index) == 1
    assert len(indices) == 2
    assert len(indices_l_band) == 1


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
    assert utils.get_new_dimension(dimension, binning_factor) == expected


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


@pytest.mark.parametrize(
    "binning_factor, expected", [(1, 128), (2, 256), (3, 512),
                                 (4, 1024), (5, 2048), (6, 4096)])
def test_upbin_image(binning_factor: int, expected: int) -> None:
    """Tests the rebinning of an image"""
    px, dim, dia = 0.1, 64, 3
    ud_image = utils.uniform_disk(px*u.mas, dim, diameter=dia*u.mas)
    rebinned_ud = utils.upbin_image(ud_image, binning_factor)

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
                f"ud_px{px}_dim{dim}_dia{dia}_upbin{binning_factor}.pdf",
                format="pdf")
    if binning_factor == 6:
        plt.show()
    plt.close()

    assert ud_image.shape == (dim, dim)
    assert rebinned_ud.shape == (expected, expected)


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
                           high_wavelength_solution: u.um,
                           wavelength_solution: u.um) -> None:
    """Tests the opacity interpolation from one wavelength
    axis to another."""
    opacity_dir = Path("opacities")
    if not opacity_dir.exists():
        opacity_dir.mkdir()

    qval_file = qval_file_dir / file
    wavelength_grid, opacity = utils.qval_to_opacity(qval_file)

    assert opacity.unit == u.cm**2/u.g
    assert opacity.shape == wavelength_grid.shape

    _, axarr = plt.subplots(2, 2, figsize=(12, 12))
    fields = [["low", "low"], ["low", "high"],
              ["high", "low"], ["high", "high"]]
    wavelength_solutions = {"low": wavelength_solution,
                            "high": high_wavelength_solution}
    opacities = []
    for field in fields:
        ind = np.where(np.logical_and(
            (wavelength_solutions[field[0]].min()-1*u.um) < wavelength_grid,
            wavelength_grid < (wavelength_solutions[field[0]].max()+1*u.um)))
        wl_grid, opc = wavelength_grid[ind], opacity[ind]
        opacities.append(utils.transform_opacity(
                wl_grid, opc, wavelength_solutions[field[0]],
                dl_coeffs=OPTIONS["spectrum.coefficients"][field[1]]))

    for index, (ax, op) in enumerate(zip(axarr.flatten(), opacities)):
        ax.plot(wl_grid.value, opc.value, label="Original")
        ax.plot(wavelength_solutions[fields[index][0]].value, op.value, label="After")
        ax.set_title(f"Wavelength Solution: {fields[index][0].upper()}; "
                     f"Coefficients: {fields[index][1].upper()}")
        ax.set_ylabel(r"$\kappa$ (cm$^{2}$g)")
        ax.set_xlabel(r"$\lambda$ (um)")
        ax.legend()
    plt.savefig(opacity_dir / f"{qval_file.stem}_kw10px.pdf", format="pdf")
    plt.close()

    assert all(opacity.unit == u.cm**2/u.g for opacity in opacities)
    for field, op in zip(fields, opacities):
        assert op.shape == wavelength_solutions[field[0]].shape


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
        high_wavelength_solution: u.um,
        wavelength_solution: u.um) -> None:
    """Tests the linear combination of interpolated wavelength grids."""
    opacity_dir = Path("opacities")
    if not opacity_dir.exists():
        opacity_dir.mkdir()

    fields = [["low", "low"], ["low", "high"],
              ["high", "low"], ["high", "high"]]
    opacities = []
    weights = np.array([42.8, 9.7, 43.5, 1.1, 2.3, 0.6])/100
    weights_background = np.append(weights, [0.668])
    files = qval_files.copy()
    files.append(qval_file_dir / "Q_SILICA_RV0.1.DAT")
    wavelength_solutions = {"low": wavelength_solution,
                            "high": high_wavelength_solution}
    for field in fields:
        opacity = []
        opacity.append(utils.linearly_combine_opacities(
            weights, qval_files,
            wavelength_solutions[field[0]], field[1]))

        opacity.append(utils.linearly_combine_opacities(
            weights_background, files,
            wavelength_solutions[field[0]], field[1]))
        opacities.append(opacity)

    _, axarr = plt.subplots(2, 2, figsize=(12, 12))
    for index, (ax, opacity) in enumerate(zip(axarr.flatten(), opacities)):
        ax.plot(wavelength_solutions[fields[index][0]].value, opacity[0].value)
        ax.set_title(f"Wavelength Solution: {fields[index][0].upper()}; "
                     f"Coefficients: {fields[index][1].upper()}")
        ax.set_xlabel(r"$\lambda$ (um)")
        ax.set_ylabel(r"$\kappa$ (cm$^{2}$g)")
    plt.savefig(opacity_dir / "combined_opacities.pdf", format="pdf")
    plt.close()

    _, axarr = plt.subplots(2, 2, figsize=(12, 12))
    for index, (ax, opacity) in enumerate(zip(axarr.flatten(), opacities)):
        ax.plot(wavelength_solutions[fields[index][0]].value, opacity[1].value)
        ax.set_title(f"Wavelength Solution: {fields[index][0].upper()}; "
                     f"Coefficients: {fields[index][1].upper()}")
        ax.set_xlabel(r"$\lambda$ (um)")
        ax.set_ylabel(r"$\kappa$ (cm$^{2}$g)")
    plt.savefig(opacity_dir / "combined_opacities_with_silica.pdf", format="pdf")
    plt.close()

    for field, opacity in zip(fields, opacities):
        assert all(op.unit == u.cm**2/u.g for op in opacity)
        assert opacity[0].shape == wavelength_solutions[field[0]].shape
        assert opacity[1].shape == wavelength_solutions[field[0]].shape


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


@pytest.mark.parametrize(
    "distance, diameter", [(2.06*u.km, 1*u.cm), (1*u.au, 725.27*u.km),
                           (1*u.lyr, 45_866_916*u.km), (1*u.pc, 1*u.au)])
def test_distance_to_angular(distance: u.Quantity,
                             diameter: u.Quantity) -> None:
    """Tests the angular diameter to diameter calculation."""
    angular_diameter = utils.distance_to_angular(diameter, distance)
    assert np.isclose(angular_diameter.to(u.arcsec), 1*u.arcsec, atol=1e-2)


def test_calculate_effective_baselines(fits_file: Path,
                                       wavelength: u.um) -> None:
    """Tests the calculation of the effective baselines."""
    axis_ratio, pos_angle = 1, 33*u.deg
    readout = ReadoutFits(fits_file)
    effective_baselines_mlambda = utils.calculate_effective_baselines(
        readout.ucoord, readout.vcoord, axis_ratio, pos_angle, wavelength)
    effective_baselines_meter = utils.calculate_effective_baselines(
        readout.ucoord, readout.vcoord, axis_ratio, pos_angle)

    assert effective_baselines_meter.unit == u.m
    assert effective_baselines_meter.size == 6
    assert effective_baselines_mlambda.unit == u.one

    effective_baselines_mlambda_cp = utils.calculate_effective_baselines(
        readout.u123coord, readout.v123coord, axis_ratio, pos_angle, wavelength)
    effective_baselines_meter_cp = utils.calculate_effective_baselines(
        readout.u123coord, readout.v123coord, axis_ratio, pos_angle)
    assert effective_baselines_meter_cp.unit == u.m
    assert effective_baselines_meter_cp.shape == (3, 4)
    assert effective_baselines_mlambda_cp.unit == u.one
