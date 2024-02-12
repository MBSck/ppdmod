from pathlib import Path
from typing import Optional, List, Tuple

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.special import j1

from ppdmod import utils
from ppdmod.data import ReadoutFits, set_fit_wavelengths, set_data
from ppdmod.options import OPTIONS


@pytest.fixture
def qval_files() -> List[Path]:
    files = ["Q_Am_Mgolivine_Jae_DHS_f1.0_rv0.1.dat",
             "Q_Am_Mgolivine_Jae_DHS_f1.0_rv1.5.dat",
             "Q_Am_Mgpyroxene_Dor_DHS_f1.0_rv1.5.dat",
             "Q_Fo_Suto_DHS_f1.0_rv0.1.dat",
             "Q_Fo_Suto_DHS_f1.0_rv1.5.dat",
             "Q_En_Jaeger_DHS_f1.0_rv1.5.dat"]
    return list(map(lambda x: Path("data/qval") / x, files))


@pytest.fixture
def grf_files() -> List[Path]:
    files = ["MgOlivine0.1.Combined.Kappa",
             "MgOlivine2.0.Combined.Kappa",
             "MgPyroxene2.0.Combined.Kappa",
             "Forsterite0.1.Combined.Kappa",
             "Forsterite2.0.Combined.Kappa",
             "Enstatite2.0.Combined.Kappa"]
    return list(map(lambda x: Path("data/grf") / x, files))


@pytest.fixture
def continuum_file() -> Path:
    return Path("data/qval") / "Q_amorph_c_rv0.1.dat"


@pytest.fixture
def fits_files() -> Path:
    """MATISSE (.fits)-files."""
    return list(Path("data/fits").glob("*2022-04-23*.fits"))


@pytest.fixture
def all_wavelength_grid() -> Path:
    """The wavelength grid."""
    data = list(map(lambda x: ReadoutFits(x).wavelength,
                    Path("data/fits").glob("*fits")))
    return np.sort(np.unique(np.concatenate(data)))


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


def calculate_effective_baselines_varga(
        ucoord: u.m, vcoord: u.m,
        compression: Optional[u.Quantity[u.one]] = None,
        pos_angle: Optional[u.Quantity[u.deg]] = None,
        longest: Optional[bool] = False) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates the effective baselines with Jozsef's approach."""
    ucoord, vcoord = map(lambda x: u.Quantity(x, u.m), [ucoord, vcoord])

    baselines_eff = np.hypot(ucoord, vcoord)
    baseline_angles_eff = np.arctan2(vcoord, ucoord)

    if pos_angle is not None:
        pos_angle = u.Quantity(pos_angle, u.deg)
        compression = u.Quantity(compression, u.one)

        diff_angle = baseline_angles_eff - pos_angle
        baselines_eff = baselines_eff*np.sqrt(
                np.cos(diff_angle)**2
                + (compression*np.sin(diff_angle))**2)

    if longest:
        indices = baselines_eff.argmax(0)
        iteration = np.arange(baselines_eff.shape[1])
        baselines_eff = baselines_eff[indices, iteration]
        diff_angle = diff_angle[indices, iteration]

    return baselines_eff.squeeze(), diff_angle.squeeze()


def calculate_effective_baselines_berger(
        ucoord: u.m, vcoord: u.m,
        compression: Optional[u.Quantity[u.one]] = None,
        pos_angle: Optional[u.Quantity[u.deg]] = None,
        longest: Optional[bool] = False) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates the effective baselines with Jozsef's approach."""
    ucoord, vcoord = map(lambda x: u.Quantity(x, u.m), [ucoord, vcoord])
    if pos_angle is not None:
        pos_angle = u.Quantity(pos_angle, u.deg)
        compression = u.Quantity(compression, u.one)

        ucoord_eff = ucoord*np.cos(pos_angle) + vcoord*np.sin(pos_angle)
        vcoord_eff = -ucoord*np.sin(pos_angle) + vcoord*np.cos(pos_angle)
        vcoord_eff *= compression
    else:
        ucoord_eff, vcoord_eff = ucoord, vcoord

    baselines_eff = np.hypot(ucoord_eff, vcoord_eff)
    baseline_angles_eff = np.arctan2(vcoord_eff, ucoord_eff)

    if longest:
        indices = baselines_eff.argmax(0)
        iteration = np.arange(baselines_eff.shape[1])
        baselines_eff = baselines_eff[indices, iteration]
        baseline_angles_eff = baseline_angles_eff[indices, iteration]

    return baselines_eff.squeeze(), baseline_angles_eff.squeeze()


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


def test_calculate_effective_baselines(
        fits_files: Path, wavelengths: u.um) -> None:
    """Tests the calculation of the effective baselines."""
    axis_ratio, pos_angle = 0.35*u.one, 33*u.deg
    fit_data = ["flux", "vis", "vis2", "t3"]
    set_fit_wavelengths(wavelengths)
    set_data(fits_files, fit_data=fit_data)

    nfiles = len(fits_files)
    baseline_dir = Path("baselines")
    if not baseline_dir.exists():
        baseline_dir.mkdir()

    vis = OPTIONS.data.vis2
    effective_baselines, baseline_angles = utils.calculate_effective_baselines(
        vis.ucoord, vis.vcoord, axis_ratio, pos_angle)

    assert effective_baselines.shape == (6*nfiles, )
    assert effective_baselines.unit == u.m
    assert baseline_angles.shape == (6*nfiles, )
    assert baseline_angles.unit == u.rad

    t3 = OPTIONS.data.t3
    effective_baselines_cp, baseline_angles = utils.calculate_effective_baselines(
        t3.u123coord, t3.v123coord, axis_ratio, pos_angle, longest=True)

    assert effective_baselines_cp.shape == (4*nfiles, )
    assert effective_baselines_cp.unit == u.m
    assert baseline_angles.shape == (4*nfiles, )
    assert baseline_angles.unit == u.rad

    _, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4))
    ax.scatter(effective_baselines.value/wavelengths.value[:, np.newaxis],
               vis.value)
    ax.set_xlabel(r"$B_{\mathrm{eff}}$ (M$\lambda$)")
    ax.set_ylabel("Visibilities")

    bx.scatter(effective_baselines_cp.value/wavelengths.value[:, np.newaxis],
               t3.value)
    bx.set_xlabel(r"$B_{\mathrm{max}}$ (M$\lambda$)")
    bx.set_ylabel("Closure phase (deg)")
    plt.savefig(baseline_dir / "effective_baselines.pdf", format="pdf")
    plt.close()


def test_compare_effective_baselines(
        fits_files: Path, wavelengths: u.um) -> None:
    """Compares the different calculations of the effective
    baseline calculation."""
    compression, pos_angle = 0.35*u.one, 33*u.deg
    fit_data = ["flux", "vis", "vis2", "t3"]
    set_fit_wavelengths(wavelengths)
    set_data(fits_files, fit_data=fit_data)

    baseline_dir = Path("baselines")
    if not baseline_dir.exists():
        baseline_dir.mkdir()

    vis, t3 = OPTIONS.data.vis2, OPTIONS.data.t3

    eff_baselines, baseline_angle = utils.calculate_effective_baselines(
        vis.ucoord, vis.vcoord, compression, pos_angle)
    eff_baselines_vg, baseline_angle_vg = calculate_effective_baselines_varga(
        vis.ucoord, vis.vcoord, compression, pos_angle)
    eff_baselines_bg, baseline_angle_bg = calculate_effective_baselines_berger(
        vis.ucoord, vis.vcoord, compression, pos_angle)

    eff_baselines_t3, baseline_angle_t3 = utils.calculate_effective_baselines(
        t3.u123coord, t3.v123coord, compression, pos_angle)
    eff_baselines_t3_vg, baseline_angle_t3_vg = calculate_effective_baselines_varga(
        t3.u123coord, t3.v123coord, compression, pos_angle)
    eff_baselines_t3_bg, baseline_angle_t3_bg = calculate_effective_baselines_berger(
        t3.u123coord, t3.v123coord, compression, pos_angle)

    _, ((ax, bx), (cx, dx)) = plt.subplots(2, 2, figsize=(12, 12))
    ax.scatter(eff_baselines.value/wavelengths.value[:, np.newaxis],
               vis.value, label="Matter")
    ax.scatter(eff_baselines_vg.value/wavelengths.value[:, np.newaxis],
               vis.value, label="Varga")
    ax.scatter(eff_baselines_bg.value/wavelengths.value[:, np.newaxis],
               vis.value, label="Berger", marker="x")
    ax.set_xlabel(r"$B_{\mathrm{eff}}$ (M$\lambda$)")
    ax.set_ylabel("Visibilities")
    ax.legend()

    bx.scatter(eff_baselines_t3.value/wavelengths.value[:, np.newaxis],
               t3.value, label="Matter")
    bx.scatter(eff_baselines_t3_vg.value/wavelengths.value[:, np.newaxis],
               t3.value, label="Varga")
    bx.scatter(eff_baselines_t3_bg.value/wavelengths.value[:, np.newaxis],
               t3.value, label="Berger", marker="x")
    bx.set_xlabel(r"$B_{\mathrm{eff}}$ (M$\lambda$)")
    bx.set_ylabel("Closure phase (deg)")
    bx.legend()

    cx.scatter(baseline_angle.value/wavelengths.value[:, np.newaxis],
               vis.value, label="Matter")
    cx.scatter(baseline_angle_vg.value/wavelengths.value[:, np.newaxis],
               vis.value, label="Varga")
    cx.scatter(baseline_angle_bg.value/wavelengths.value[:, np.newaxis],
               vis.value, label="Berger", marker="x")
    cx.set_xlabel(r"$\phi_{\mathrm{eff}}$ (deg)")
    cx.set_ylabel("Visibilities")
    cx.legend()

    dx.scatter(baseline_angle_t3.value/wavelengths.value[:, np.newaxis],
               t3.value, label="Matter")
    dx.scatter(baseline_angle_t3_vg.value/wavelengths.value[:, np.newaxis],
               t3.value, label="Varga")
    dx.scatter(baseline_angle_t3_bg.value/wavelengths.value[:, np.newaxis],
               t3.value, label="Berger", marker="x")
    dx.set_xlabel(r"$\phi_{\mathrm{eff}}$ (deg)")
    dx.set_ylabel("Closure phase (deg)")
    dx.legend()
    plt.savefig(baseline_dir / "baseline_comparison.pdf", format="pdf")
    plt.close()

    diameter, wavelength = 20*u.mas, 10*u.um
    xx = np.linspace(-0.5, 0.5, 100)*500
    xx, yy = np.meshgrid(xx, xx)*u.m
    angles = np.arctan2(yy, xx)
    ud_vis = utils.uniform_disk_vis(diameter, xx, yy, wavelength)

    baselines_matter, angle_matter = utils.calculate_effective_baselines(
            xx, yy, compression, pos_angle)
    baselines_vg, angle_vg = calculate_effective_baselines_varga(
            xx, yy, compression, pos_angle)
    baselines_bg, angle_bg = calculate_effective_baselines_berger(
            xx, yy, compression, pos_angle)

    eff_bg = baselines_matter/wavelength.to(u.m)
    ud_vis_matter = 2*j1(np.pi*eff_bg*diameter.to(u.rad).value)\
        / (np.pi*diameter.to(u.rad).value*eff_bg)

    eff_vg = baselines_vg/wavelength.to(u.m)
    ud_vis_vg = 2*j1(np.pi*eff_vg*diameter.to(u.rad).value)\
        / (np.pi*diameter.to(u.rad).value*eff_vg)

    eff_bg = baselines_bg/wavelength.to(u.m)
    ud_vis_bg = 2*j1(np.pi*eff_bg*diameter.to(u.rad).value)\
        / (np.pi*diameter.to(u.rad).value*eff_bg)

    images = [ud_vis, ud_vis_matter, ud_vis_vg, ud_vis_bg]
    angles = [angles, angle_matter, angle_vg, angle_bg]
    names = ["Uniform Disk", "Matter", "Varga", "Berger"]

    extent = [xx.min().value, xx.max().value,
              yy.min().value, yy.max().value]
    _, axarr = plt.subplots(2, 4, figsize=(14, 8),
                            sharey=True, sharex=True)
    for index, (vis, angle, name) in enumerate(zip(images, angles, names)):
        ax, bx = axarr[:, index]
        ax.imshow(np.abs(vis.value), extent=extent)
        ax.set_title(name)
        if index == 0:
            ax.set_ylabel("v (m)")
            bx.set_ylabel("v (m)")
        bx.imshow(angle.value, extent=extent)
        bx.set_xlabel("u (m)")

    plt.savefig(baseline_dir / "deprojection_comparison.pdf", format="pdf")
    plt.close()


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
    uniform_disk = utils.uniform_disk(
            0.1*u.mas, 512, diameter=4*u.mas)
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


def test_qval_to_opacity(qval_files: List[Path]) -> None:
    """Tests the conversion of Q-values to opacities."""
    wavelength, opacity = utils.qval_to_opacity(qval_files[0])
    assert wavelength.unit == u.um
    assert opacity.unit == u.cm**2/u.g


def test_restrict_wavelength() -> None:
    """Tests the restriction of the wavelength grid."""
    wavelength = np.linspace(0, 245, 100, endpoint=False)
    array = np.arange(0, 100)
    indices = (wavelength > 10) & (wavelength < 50)
    new_wavelength, new_array = utils.restrict_wavelength(
            wavelength, array, [10, 50])
    assert new_wavelength.size == wavelength[indices].size
    assert new_array.size == array[indices].size


@pytest.mark.parametrize("shape", [(1, 10), (2, 10), (3, 10)])
def test_restrict_phase(shape: Tuple[int, int]) -> None:
    """Tests the restriction of the phase grid."""
    phases = np.random.rand(*shape)*360
    new_phases = utils.restrict_phase(phases)
    assert ((phases > 180) | (phases < -180)).any()
    assert ((new_phases < 180) | (new_phases > -180)).all()


def test_load_data(qval_files: List[Path],
                   grf_files: List[Path]) -> None:
    """Tests the loading of a data file."""
    wavelength_grids, data = utils.load_data(
            qval_files[0], load_func=utils.qval_to_opacity)
    assert len(data.shape) == 1
    assert len(wavelength_grids.shape) == 1

    wavelength_grids, data = utils.load_data(
            qval_files, load_func=utils.qval_to_opacity)
    assert data.shape[0] == 6
    assert wavelength_grids.shape[0] == 6

    wavelength_grids, data = utils.load_data(grf_files)
    assert data.shape[0] == 6
    assert wavelength_grids.shape[0] == 6


def test_linearly_combine_data(
        qval_files: List[Path],
        grf_files: List[Path],
        continuum_file: Path,
        high_wavelength_solution: u.um,
        wavelength_solutions: u.um,
        all_wavelength_grid: u.um) -> None:
    """Tests the linear combination of interpolated wavelength grids."""
    low_wavelength_solution = wavelength_solutions[1].value
    high_wavelength_solution = high_wavelength_solution.value
    all_wavelength_grid = all_wavelength_grid.value

    opacity_dir = Path("opacities")
    if not opacity_dir.exists():
        opacity_dir.mkdir()

    weights = np.array([42.8, 9.7, 43.5, 1.1, 2.3, 0.6])/100
    wavelength_qval, qval = utils.load_data(
            qval_files, load_func=utils.qval_to_opacity)
    wavelength_qval = wavelength_qval[0]
    qval_combined = utils.linearly_combine_data(qval, weights)

    wavelength_grf, grf = utils.load_data(grf_files)
    wavelength_grf = wavelength_grf[0]
    grf_combined = utils.linearly_combine_data(grf, weights)

    assert (grf.shape[1],) == grf_combined.shape
    assert (qval.shape[1],) == qval_combined.shape

    wavelength_cont, continuum_data = utils.load_data(
            continuum_file, load_func=utils.qval_to_opacity)

    cont_low = np.interp(
            low_wavelength_solution, wavelength_cont, continuum_data)
    cont_high = np.interp(
            high_wavelength_solution, wavelength_cont, continuum_data)
    qval_low = np.interp(low_wavelength_solution, wavelength_qval, qval_combined)
    qval_high = np.interp(high_wavelength_solution, wavelength_qval, qval_combined)
    grf_low = np.interp(low_wavelength_solution, wavelength_grf, grf_combined)
    grf_high = np.interp(high_wavelength_solution, wavelength_grf, grf_combined)

    cont_ind = (wavelength_cont >= 1)\
        & (wavelength_cont <= low_wavelength_solution[-1])
    qval_ind = (wavelength_qval >= 1)\
        & (wavelength_qval <= low_wavelength_solution[-1])
    grf_ind = (wavelength_grf >= 1)\
        & (wavelength_grf <= low_wavelength_solution[-1])

    _, (ax, bx, cx) = plt.subplots(1, 3, figsize=(12, 4))
    ax.plot(wavelength_qval[qval_ind], qval_combined[qval_ind])
    ax.plot(wavelength_grf[grf_ind], grf_combined[grf_ind])
    ax.plot(wavelength_cont[cont_ind], continuum_data[cont_ind])
    ax.set_title("No Intp")
    ax.set_xlabel(r"$\lambda$ ($\mu$m)")
    ax.set_ylabel(r"$\kappa$ ($cm^{2}g^{-1}$)")

    bx.plot(low_wavelength_solution, qval_low)
    bx.plot(low_wavelength_solution, grf_low)
    bx.plot(low_wavelength_solution, cont_low)
    bx.set_title("Intp LOW")
    bx.set_xlabel(r"$\lambda$ ($\mu$m)")

    cx.plot(high_wavelength_solution, qval_high, label="Silicates (DHS)")
    cx.plot(high_wavelength_solution, grf_high, label="Silicates (GRF)")
    cx.plot(high_wavelength_solution, cont_high, label="Continuum (DHS)")
    cx.set_title("Intp HIGH")
    cx.set_xlabel(r"$\lambda$ ($\mu$m)")
    cx.legend()
    plt.savefig(opacity_dir / "combined_opacities.pdf", format="pdf")
    plt.close()

    cont_intp = np.interp(
            all_wavelength_grid, wavelength_cont, continuum_data)
    qval_intp = np.interp(all_wavelength_grid, wavelength_qval, qval_combined)
    grf_intp = np.interp(all_wavelength_grid, wavelength_grf, grf_combined)
    cont_ind = (wavelength_cont >= all_wavelength_grid[0])\
        & (wavelength_cont <= all_wavelength_grid[-1])
    qval_ind = (wavelength_qval >= 1)\
        & (wavelength_qval <= all_wavelength_grid[-1])
    grf_ind = (wavelength_grf >= 1)\
        & (wavelength_grf <= all_wavelength_grid[-1])

    _, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4))
    ax.plot(wavelength_qval[qval_ind], qval_combined[qval_ind])
    ax.plot(wavelength_grf[grf_ind], grf_combined[grf_ind])
    ax.plot(wavelength_cont[cont_ind], continuum_data[cont_ind])
    ax.set_title("No Interpolation")
    ax.set_xlabel(r"$\lambda$ ($\mu$m)")
    ax.set_ylim([None, 5000])

    bx.plot(all_wavelength_grid, qval_intp, label="Silicates (DHS)")
    bx.plot(all_wavelength_grid, grf_intp, label="Silicates (GRF)")
    bx.plot(all_wavelength_grid, cont_intp, label="Continuum")
    bx.set_title("Interpolation")
    bx.set_xlabel(r"$\lambda$ ($\mu$m)")
    bx.set_ylabel(r"$\kappa$ ($cm^{2}g^{-1}$)")
    bx.set_ylim([None, 5000])
    bx.legend()
    plt.savefig(opacity_dir / "combined_opacities2.pdf", format="pdf")
    plt.close()
