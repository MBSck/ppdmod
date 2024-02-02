from pathlib import Path
from typing import List

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ppdmod import utils
from ppdmod.data import ReadoutFits
from ppdmod.options import OPTIONS

# TODO: Finish the tests here


@pytest.fixture
def qval_dir() -> Path:
    """The qval-file directory."""
    return Path("data/qval")


@pytest.fixture
def qval_files(qval_dir: Path) -> List[Path]:
    files = ["Q_Am_Mgolivine_Jae_DHS_f1.0_rv0.1.dat",
             "Q_Am_Mgolivine_Jae_DHS_f1.0_rv1.5.dat",
             "Q_Am_Mgpyroxene_Dor_DHS_f1.0_rv1.5.dat",
             "Q_Fo_Suto_DHS_f1.0_rv0.1.dat",
             "Q_Fo_Suto_DHS_f1.0_rv1.5.dat",
             "Q_En_Jaeger_DHS_f1.0_rv1.5.dat"]
    return list(map(lambda x: qval_dir / x, files))


@pytest.fixture
def grf_files(grf_dir: Path) -> List[Path]:
    files = ["MgOlivine0.1.Combined.Kappa",
             "MgOlivine2.0.Combined.Kappa",
             "MgPyroxene2.0.Combined.Kappa",
             "Forsterite0.1.Combined.Kappa",
             "Forsterite2.0.Combined.Kappa",
             "Enstatite2.0.Combined.Kappa"]
    return list(map(lambda x: grf_dir / x, files))


@pytest.fixture
def continuum_file(qval_dir: Path) -> Path:
    return qval_dir / "Q_amorph_c_rv0.1.dat"


@pytest.fixture
def fits_files() -> Path:
    """MATISSE (.fits)-files."""
    return list(Path("data/fits").glob("*2022-04-23*.fits"))


@pytest.fixture
def wavelengths() -> u.um:
    """A wavelength grid."""
    return [3.520375, 8.502626, 10.001093]*u.um


@pytest.fixture
def wavelength_solutions(fits_files: Path) -> u.um:
    """The wavelength solutions for low resolution MATISSE
    (.fits)-files."""
    return [ReadoutFits(fits_file).wavelength for fits_file in fits_files]


@pytest.fixture
def high_wavelength_solution() -> u.um:
    """The wavelength solution of high resolution MATISSE (.fits)-file."""
    return np.load(Path("data") / "high_wavelength_solution.npy")*u.um


def test_exection_time() -> None:
    ...


def test_make_workbook() -> None:
    ...


def test_get_closest_indices(wavelengths: u.um,
                             wavelength_solutions: u.um) -> None:
    """Tests the get_closest_indices function."""
    index = utils.get_closest_indices(
            wavelengths[0], array=wavelength_solutions[0])
    assert len(index) == 1

    for window in [None, 0.1]:
        indices_l_band = utils.get_closest_indices(
            wavelengths, array=wavelength_solutions[0], window=window)
        indices_n_band = utils.get_closest_indices(
                wavelengths, array=wavelength_solutions[1], window=window)
        assert len(indices_l_band) == 3
        assert len(indices_n_band) == 3
        assert len([i for i in indices_l_band if i.size != 0]) == 1
        assert len([i for i in indices_n_band if i.size != 0]) == 2



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


# TODO: Check if it works
# TODO: Make sure it calculates the longest baselines as well
# TODO: Test it with multiple files (GRAVITY+PIONIER+MATISSE)
# def test_calculate_effective_baselines(fits_files: Path,
#                                        wavelengths: u.um) -> None:
#     """Tests the calculation of the effective baselines."""
#     axis_ratio, pos_angle = 0.35, 33*u.deg
#     wavelength = wavelengths[0]
#     readouts = [ReadoutFits(fits_file) for fits_file in fits_files]
#     ucoord = np.concatenate([readout.ucoord for readout in readouts])
#     vcoord = np.concatenate([readout.vcoord for readout in readouts])
#     breakpoint()

#     baseline_dir = Path("baselines")
#     if not baseline_dir.exists():
#         baseline_dir.mkdir()

#     index = np.where(wavelength == readout.wavelength)

#     effective_baselines, baseline_angles = utils.calculate_effective_baselines(
#         readout.ucoord, readout.vcoord, axis_ratio, pos_angle)

#     plt.scatter(effective_baselines/wavelength.value,
#                 readout.vis[:, index].squeeze())
#     plt.savefig(baseline_dir / "baseline_vs_vis.pdf", format="pdf")
#     plt.close()

#     assert effective_baselines.size == 6
#     assert effective_baselines.unit == u.m
#     assert baseline_angles.size == 6
#     assert baseline_angles.unit == u.rad

#     effective_baselines_cp, baseline_angles = utils.calculate_effective_baselines(
#         readout.u123coord, readout.v123coord, axis_ratio, pos_angle)
#     baseline_angles = baseline_angles
#     breakpoint()
#     effective_baselines_cp = effective_baselines.max(axis=0)

#     plt.scatter(effective_baselines_cp/wavelength.value,
#                 readout.t3[:, index].squeeze())
#     plt.savefig(baseline_dir / "baseline_vs_t3.pdf", format="pdf")
#     plt.close()

#     assert effective_baselines_cp.shape == (4,)
#     assert effective_baselines_cp.unit == u.m
#     assert baseline_angles.shape == (4,)


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


def test_binary_vis(wavelengths: u.um) -> None:
    """Tests the calculation of the binary's complex
    visibility function."""
    ucoord = np.linspace(80, 100, 20)*u.m
    flux1, flux2 = 5*u.Jy, 2*u.Jy
    position1, position2 = [5, 10]*u.mas, [-10, -10]*u.mas
    binary_vis = utils.binary_vis(
            flux1, flux2, ucoord, ucoord*0,
            position1, position2, wavelengths[0])
    assert isinstance(binary_vis, np.ndarray)
    assert not np.array_equal(np.real(binary_vis), binary_vis)


def test_uniform_disk() -> None:
    """Tests the calculation of a uniform disk's brightness."""
    uniform_disk = utils.uniform_disk(0.1*u.mas, 512,
                                      diameter=4*u.mas)
    assert uniform_disk.shape == (512, 512)
    assert uniform_disk.unit == u.one


def test_uniform_disk_vis(wavelengths: u.um) -> None:
    """Tests the calculation of a uniform disk's complex
    visibility function."""
    ucoord = np.linspace(80, 100, 20)*u.m
    uniform_disk_vis = utils.uniform_disk_vis(
            4*u.mas, ucoord, ucoord*0, wavelengths[0])
    assert uniform_disk_vis.unit == u.one
    assert np.array_equal(np.real(uniform_disk_vis), uniform_disk_vis)


# @pytest.mark.parametrize(
#     "file", ["Q_Am_Mgolivine_Jae_DHS_f1.0_rv0.1.dat",
#              "Q_Am_Mgolivine_Jae_DHS_f1.0_rv1.5.dat",
#              "Q_Am_Mgpyroxene_Dor_DHS_f1.0_rv1.5.dat",
#              "Q_Fo_Suto_DHS_f1.0_rv0.1.dat",
#              "Q_Fo_Suto_DHS_f1.0_rv1.5.dat",
#              "Q_En_Jaeger_DHS_f1.0_rv1.5.dat"])
# def test_transform_data(qval_file_dir: Path, file: str,
#                         high_wavelength_solution: u.um,
#                         wavelength_solution: u.um) -> None:
#     """Tests the opacity interpolation from one wavelength
#     axis to another."""
#     opacity_dir = Path("opacities")
#     if not opacity_dir.exists():
#         opacity_dir.mkdir()

#     qval_file = qval_file_dir / file
#     wavelength_grid, opacity = utils.qval_to_opacity(qval_file)

#     assert opacity.unit == u.cm**2/u.g
#     assert opacity.shape == wavelength_grid.shape

#     _, axarr = plt.subplots(2, 2, figsize=(12, 12))
#     fields = [["low", "low"], ["low", "high"],
#               ["high", "low"], ["high", "high"]]
#     wavelength_solutions = {"low": wavelength_solution,
#                             "high": high_wavelength_solution}
#     opacities = []
#     for field in fields:
#         ind = np.where(np.logical_and(
#             (wavelength_solutions[field[0]].min()-1*u.um) < wavelength_grid,
#             wavelength_grid < (wavelength_solutions[field[0]].max()+1*u.um)))
#         wl_grid, opc = wavelength_grid[ind], opacity[ind]
#         dl_coeffs = getattr(OPTIONS.spectrum.coefficients, field[1])
#         opacities.append(utils.transform_data(
#                 wl_grid, opc, wavelength_solutions[field[0]], dl_coeffs=dl_coeffs))

#     for index, (ax, op) in enumerate(zip(axarr.flatten(), opacities)):
#         ax.plot(wl_grid.value, opc.value, label="Original")
#         ax.plot(wavelength_solutions[fields[index][0]].value, op.value, label="After")
#         ax.set_title(f"Wavelength Solution: {fields[index][0].upper()}; "
#                      f"Coefficients: {fields[index][1].upper()}")
#         ax.set_ylabel(r"$\kappa$ (cm$^{2}$g)")
#         ax.set_xlabel(r"$\lambda$ (um)")
#         ax.legend()
#     plt.savefig(opacity_dir / f"{qval_file.stem}_kw10px.pdf", format="pdf")
#     plt.close()

#     assert all(opacity.unit == u.cm**2/u.g for opacity in opacities)
#     for field, op in zip(fields, opacities):
#         assert op.shape == wavelength_solutions[field[0]].shape


# # NOTE: This test, tests nothing
# # TODO: Make a plot here
# def test_opacity_to_matisse_opacity(
#         qval_file_dir: Path, wavelength_solution: u.um) -> None:
#     """Tests the interpolation to the MATISSE wavelength grid."""
#     qval_file = qval_file_dir / "Q_SILICA_RV0.1.DAT"
#     continuum_opacity = utils.data_to_matisse_grid(wavelength_solution,
#                                                    qval_file=qval_file)
#     assert continuum_opacity.unit == u.cm**2/u.g


# # TODO: Check if it is combined properly.
# def test_linearly_combine_opacities(
#         qval_file_dir: Path,
#         qval_files: List[Path],
#         high_wavelength_solution: u.um,
#         wavelength_solution: u.um) -> None:
#     """Tests the linear combination of interpolated wavelength grids."""
#     opacity_dir = Path("opacities")
#     if not opacity_dir.exists():
#         opacity_dir.mkdir()

#     fields = [["low", "low"], ["low", "high"],
#               ["high", "low"], ["high", "high"]]
#     opacities = []
#     weights = np.array([42.8, 9.7, 43.5, 1.1, 2.3, 0.6])/100
#     weights_background = np.append(weights, [0.668])
#     files = qval_files.copy()
#     files.append(qval_file_dir / "Q_SILICA_RV0.1.DAT")
#     wavelength_solutions = {"low": wavelength_solution,
#                             "high": high_wavelength_solution}
#     for field in fields:
#         opacity = []
#         opacity.append(utils.linearly_combine_data(
#             weights, qval_files,
#             wavelength_solutions[field[0]], field[1]))

#         opacity.append(utils.linearly_combine_data(
#             weights_background, files,
#             wavelength_solutions[field[0]], field[1]))
#         opacities.append(opacity)

#     _, axarr = plt.subplots(2, 2, figsize=(12, 12))
#     for index, (ax, opacity) in enumerate(zip(axarr.flatten(), opacities)):
#         ax.plot(wavelength_solutions[fields[index][0]].value, opacity[0].value)
#         ax.set_title(f"Wavelength Solution: {fields[index][0].upper()}; "
#                      f"Coefficients: {fields[index][1].upper()}")
#         ax.set_xlabel(r"$\lambda$ (um)")
#         ax.set_ylabel(r"$\kappa$ (cm$^{2}$g)")
#     plt.savefig(opacity_dir / "combined_opacities.pdf", format="pdf")
#     plt.close()

#     _, axarr = plt.subplots(2, 2, figsize=(12, 12))
#     for index, (ax, opacity) in enumerate(zip(axarr.flatten(), opacities)):
#         ax.plot(wavelength_solutions[fields[index][0]].value, opacity[1].value)
#         ax.set_title(f"Wavelength Solution: {fields[index][0].upper()}; "
#                      f"Coefficients: {fields[index][1].upper()}")
#         ax.set_xlabel(r"$\lambda$ (um)")
#         ax.set_ylabel(r"$\kappa$ (cm$^{2}$g)")
#     plt.savefig(opacity_dir / "combined_opacities_with_silica.pdf", format="pdf")
#     plt.close()

#     for field, opacity in zip(fields, opacities):
#         assert all(op.unit == u.cm**2/u.g for op in opacity)
#         assert opacity[0].shape == wavelength_solutions[field[0]].shape
#         assert opacity[1].shape == wavelength_solutions[field[0]].shape


