from pathlib import Path
from typing import List, Optional, Tuple

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.special import j1

from ppdmod import utils
from ppdmod.basic_components import Gaussian, PointSource, Star, TempGradient
from ppdmod.component import FourierComponent
from ppdmod.data import ReadoutFits, get_all_wavelengths, set_data
from ppdmod.options import OPTIONS, STANDARD_PARAMS
from ppdmod.parameter import Parameter

DATA_DIR = Path(__file__).parent.parent / "data"

@pytest.fixture
def qval_files() -> List[Path]:
    files = [
        "Q_Am_Mgolivine_Jae_DHS_f1.0_rv0.1.dat",
        "Q_Am_Mgolivine_Jae_DHS_f1.0_rv1.5.dat",
        "Q_Am_Mgpyroxene_Dor_DHS_f1.0_rv1.5.dat",
        "Q_Fo_Suto_DHS_f1.0_rv0.1.dat",
        "Q_Fo_Suto_DHS_f1.0_rv1.5.dat",
        "Q_En_Jaeger_DHS_f1.0_rv1.5.dat",
    ]
    return list(map(lambda x: DATA_DIR / "opacities" / "qval" / x, files))


@pytest.fixture
def grf_files() -> List[Path]:
    files = [
        "MgOlivine0.1.Combined.Kappa",
        "MgOlivine2.0.Combined.Kappa",
        "MgPyroxene2.0.Combined.Kappa",
        "Forsterite0.1.Combined.Kappa",
        "Forsterite2.0.Combined.Kappa",
        "Enstatite2.0.Combined.Kappa",
    ]
    return list(map(lambda x: DATA_DIR / "opacities" / "grf" / x, files))


@pytest.fixture
def continuum_file() -> Path:
    return DATA_DIR / "opacities" / "optool" / "preibisch_amorph_c_rv0.1.npy"


@pytest.fixture
def fits_files() -> Path:
    """MATISSE (.fits)-files."""
    return list((DATA_DIR / "matisse").glob("*.fits"))


@pytest.fixture
def all_wavelength_grid(fits_files) -> Path:
    """The wavelength grid."""
    set_data(fits_files, wavelengths="all", fit_data=["vis", "vis2", "t3"])
    return get_all_wavelengths()


@pytest.fixture
def wavelengths() -> u.um:
    """A wavelength grid."""
    return [3.5, 8.5, 10] * u.um


@pytest.fixture
def wavelength_solutions(fits_files: Path) -> u.um:
    """The wavelength solutions for low resolution MATISSE
    (.fits)-files."""
    return [ReadoutFits(fits_file).wavelength for fits_file in fits_files]


@pytest.fixture
def high_wavelength_solution() -> u.um:
    """The wavelength solution of high resolution MATISSE (.fits)-file."""
    return np.load(DATA_DIR / "wl_grids" / "fits" / "matisse" / "nband_high.npy") * u.um


def uniform_disk_vis(baselines, diameter):
    return (
        2
        * j1(np.pi * baselines * diameter.to(u.rad).value)
        / (np.pi * diameter.to(u.rad).value * baselines)
    )


def calculate_effective_baselines_varga(
    ucoord: u.m,
    vcoord: u.m,
    compression: Optional[u.Quantity[u.one]] = None,
    pos_angle: Optional[u.Quantity[u.deg]] = None,
    longest: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates the effective baselines with Jozsef's approach."""
    ucoord, vcoord = map(lambda x: u.Quantity(x, u.m), [ucoord, vcoord])

    baselines_eff = np.hypot(ucoord, vcoord)
    diff_angle = np.arctan2(ucoord, vcoord)

    if pos_angle is not None:
        pos_angle = u.Quantity(pos_angle, u.deg)
        compression = u.Quantity(compression, u.one)

        diff_angle -= pos_angle
        baselines_eff = baselines_eff * np.sqrt(
            np.cos(diff_angle) ** 2 + (compression * np.sin(diff_angle)) ** 2
        )

    if longest:
        indices = baselines_eff.argmax(0)
        iteration = np.arange(baselines_eff.shape[1])
        baselines_eff = baselines_eff[indices, iteration]
        diff_angle = diff_angle[indices, iteration]

    return baselines_eff.squeeze(), diff_angle.squeeze()


def calculate_effective_baselines_berger(
    ucoord: u.m,
    vcoord: u.m,
    compression: Optional[u.Quantity[u.one]] = None,
    pos_angle: Optional[u.Quantity[u.deg]] = None,
    longest: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates the effective baselines with Jozsef's approach."""
    ucoord, vcoord = map(lambda x: u.Quantity(x, u.m), [ucoord, vcoord])
    if pos_angle is not None:
        pos_angle = u.Quantity(pos_angle, u.deg)
        compression = u.Quantity(compression, u.one)

        ucoord_eff = ucoord * np.cos(pos_angle) + vcoord * np.sin(pos_angle)
        vcoord_eff = -ucoord * np.sin(pos_angle) + vcoord * np.cos(pos_angle)
    else:
        ucoord_eff, vcoord_eff = ucoord, vcoord

    if compression is not None:
        vcoord_eff *= compression

    baselines_eff = np.hypot(ucoord_eff, vcoord_eff)
    baseline_angles_eff = np.arctan2(ucoord_eff, vcoord_eff)

    if longest:
        indices = baselines_eff.argmax(0)
        iteration = np.arange(baselines_eff.shape[1])
        baselines_eff = baselines_eff[indices, iteration]
        baseline_angles_eff = baseline_angles_eff[indices, iteration]

    return baselines_eff.squeeze(), baseline_angles_eff.squeeze()


def test_exection_time() -> None: ...


def test_make_workbook() -> None: ...


def test_get_indices(wavelengths: u.um, wavelength_solutions: u.um) -> None:
    """Tests the get_closest_indices function."""
    index = utils.get_indices(
        wavelengths[0],
        array=wavelength_solutions[0],
        window=OPTIONS.data.binning.unknown,
    )
    assert index[0].size >= 1

    indices_lband = utils.get_indices(
        wavelengths, array=wavelength_solutions[0], window=0.1
    )
    indices_nband = utils.get_indices(
        wavelengths, array=wavelength_solutions[2], window=0.1
    )
    assert len(indices_lband) == 3
    assert len(indices_nband) == 3
    len_nband = len([i for i in indices_lband if i.size != 0])
    len_lband = len([i for i in indices_nband if i.size != 0])
    assert len_nband == 1
    assert len_lband == 2


@pytest.mark.parametrize(
    "distance, expected",
    [
        (2.06 * u.km, 1 * u.cm),
        (1 * u.au, 725.27 * u.km),
        (1 * u.lyr, 45_866_916 * u.km),
        (1 * u.pc, 1 * u.au),
    ],
)
def test_angular_to_distance(distance: u.Quantity, expected: u.Quantity) -> None:
    """Tests the angular diameter equation to
    convert angules to distance.

    Test values for 1" angular diameter from wikipedia:
    https://en.wikipedia.org/wiki/Angular_diameter.
    """
    diameter = utils.angular_to_distance(1 * u.arcsec, distance)
    assert np.isclose(diameter, expected.to(u.m), atol=1e-3)


@pytest.mark.parametrize(
    "distance, diameter",
    [
        (2.06 * u.km, 1 * u.cm),
        (1 * u.au, 725.27 * u.km),
        (1 * u.lyr, 45_866_916 * u.km),
        (1 * u.pc, 1 * u.au),
    ],
)
def test_distance_to_angular(distance: u.Quantity, diameter: u.Quantity) -> None:
    """Tests the angular diameter to diameter calculation."""
    angular_diameter = utils.distance_to_angular(diameter, distance)
    assert np.isclose(angular_diameter.to(u.arcsec), 1 * u.arcsec, atol=1e-2)


def test_calculate_effective_baselines(fits_files: Path, wavelengths: u.um) -> None:
    """Tests the calculation of the effective baselines."""
    axis_ratio, pos_angle = 2.85 * u.one, 33 * u.deg
    set_data(
        fits_files, wavelengths=wavelengths, fit_data=["flux", "vis", "vis2", "t3"]
    )

    nfiles = len(fits_files)
    baseline_dir = Path("baselines")
    if not baseline_dir.exists():
        baseline_dir.mkdir()

    vis = OPTIONS.data.vis2
    effective_baselines, baseline_angles = utils.compute_effective_baselines(
        vis.ucoord, vis.vcoord, axis_ratio, pos_angle
    )

    assert effective_baselines.shape == (6 * nfiles + 1,)
    assert effective_baselines.unit == u.m
    assert baseline_angles.shape == (6 * nfiles + 1,)
    assert baseline_angles.unit == u.rad

    t3 = OPTIONS.data.t3
    effective_baselines_cp, baseline_angles = utils.compute_effective_baselines(
        t3.u123coord, t3.v123coord, axis_ratio, pos_angle, longest=True
    )

    assert effective_baselines_cp.shape == (4 * nfiles + 1,)
    assert effective_baselines_cp.unit == u.m
    assert baseline_angles.shape == (4 * nfiles + 1,)
    assert baseline_angles.unit == u.rad

    wavelengths = wavelengths.value[:, np.newaxis]
    _, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4))
    ax.scatter((effective_baselines.value / wavelengths)[:, 1:], vis.value)
    ax.set_xlabel(r"$B_{\mathrm{eff}}$ (M$\lambda$)")
    ax.set_ylabel("Visibilities")

    bx.scatter((effective_baselines_cp.value / wavelengths)[:, 1:], t3.value)
    bx.set_xlabel(r"$B_{\mathrm{max}}$ (M$\lambda$)")
    bx.set_ylabel("Closure phase (deg)")
    plt.savefig(baseline_dir / "effective_baselines.pdf", format="pdf")

    plt.close()
    set_data(fit_data=["flux", "vis", "vis2", "t3"])


# TODO: Test this also for the angles, but it seems promising
@pytest.mark.parametrize(
    "fits_file, compression, pos_angle",
    [
        ("ud.fits", None, None),
        ("ud_inc.fits", 0.351, None),
        ("ud_inc_rot.fits", 0.351, 33 * u.deg),
    ],
)
def test_compare_effective_baselines(
    fits_file: Path, compression: float, pos_angle: u.deg
) -> None:
    """Compares the different calculations of the effective
    baseline calculation."""
    fits_file = DATA_DIR / "aspro" / fits_file
    diameter, wavelength = 20 * u.mas, [10] * u.um
    set_data([fits_file], wavelengths=wavelength, fit_data=["vis", "t3"])

    baseline_dir = Path("baselines")
    baseline_dir.mkdir(exist_ok=True, parents=True)

    vis = OPTIONS.data.vis
    calc_func = [
        utils.compute_effective_baselines,
        calculate_effective_baselines_varga,
        calculate_effective_baselines_berger,
    ]

    names = ["Aspro", "Corr. Matter", "Varga", "Corr. Berger"]
    ud_visibilities = [vis.value]
    all_baselines = [np.hypot(vis.ucoord, vis.vcoord) * u.m / wavelength.to(u.m)]
    all_angles = [np.arctan2(vis.vcoord, vis.ucoord) * u.rad]
    for func in calc_func:
        baselines, angles = func(vis.ucoord, vis.vcoord, compression, pos_angle)
        baselines = baselines / wavelength.to(u.m)
        all_baselines.append(baselines)
        all_angles.append(angles)
        ud_visibilities.append(uniform_disk_vis(baselines, diameter))

    assert np.allclose(vis.value, ud_visibilities[0], atol=1e-2)

    _, (ax, bx) = plt.subplots(1, 2, figsize=(12, 6))
    for name, baseline, angle, ud_vis in zip(
        names, all_baselines, all_angles, ud_visibilities
    ):
        baseline = baseline.value.squeeze() * 1e-6
        baseline_angle = angle.to(u.deg).value.squeeze()
        if ud_vis.size != baseline.size:
            baseline, baseline_angle = baseline[1:], baseline_angle[1:]

        ax.scatter(baseline, ud_vis, label=name, alpha=0.6)
        bx.scatter(baseline_angle, ud_vis, label=name, alpha=0.6)

    ax.set_xlabel(r"$B_{\mathrm{eff}}$ (M$\lambda$)")
    ax.set_ylabel("Visibilities")
    ax.legend()

    bx.set_xlabel(r"$\phi_{\mathrm{eff}}$ (deg)")
    bx.set_ylabel("Visibilities")
    bx.legend()

    title = "Uniform Disk"
    if "inc" in fits_file.name:
        title += " Inclined"
    if "rot" in fits_file.name:
        title += ", Rotated"

    plt.savefig(baseline_dir / f"deprojection_{fits_file.name}.pdf", format="pdf")
    plt.close()

    set_data(fit_data=["vis", "t3"])


def test_binary() -> None:
    """Tests the calculation of a binary's brightness."""
    flux1, flux2 = 5 * u.Jy, 2 * u.Jy
    position1, position2 = [5, 10] * u.mas, [-10, -10] * u.mas
    binary = utils.binary(512, 0.1 * u.mas, flux1, flux2, position1, position2)
    assert binary[binary != 0].size == 2
    assert binary.shape == (512, 512)
    assert binary.unit == u.Jy


def test_binary_vis(wavelengths: u.um) -> None:
    """Tests the calculation of the binary's complex
    visibility function."""
    ucoord = np.linspace(80, 100, 20) * u.m
    flux1, flux2 = 5 * u.Jy, 2 * u.Jy
    position1, position2 = [5, 10] * u.mas, [-10, -10] * u.mas
    binary_vis = utils.binary_vis(
        flux1, flux2, ucoord, ucoord * 0, position1, position2, wavelengths[0]
    )
    assert isinstance(binary_vis, np.ndarray)
    assert not np.array_equal(np.real(binary_vis), binary_vis)


def test_uniform_disk() -> None:
    """Tests the calculation of a uniform disk's brightness."""
    uniform_disk = utils.uniform_disk(0.1 * u.mas, 512, diameter=4 * u.mas)
    assert uniform_disk.shape == (512, 512)
    assert uniform_disk.unit == u.one


def test_uniform_disk_vis(wavelengths: u.um) -> None:
    """Tests the calculation of a uniform disk's complex
    visibility function."""
    ucoord = np.linspace(80, 100, 20) * u.m
    uniform_disk_vis = utils.uniform_disk_vis(
        4 * u.mas, ucoord, ucoord * 0, wavelengths[0]
    )
    assert uniform_disk_vis.unit == u.one
    assert np.array_equal(np.real(uniform_disk_vis), uniform_disk_vis)


def test_qval_to_opacity(qval_files: List[Path]) -> None:
    """Tests the conversion of Q-values to opacities."""
    wavelength, opacity = utils.qval_to_opacity(qval_files[0])
    assert wavelength.unit == u.um
    assert opacity.unit == u.cm**2 / u.g


def test_restrict_wavelength() -> None:
    """Tests the restriction of the wavelength grid."""
    wavelength = np.linspace(0, 245, 100, endpoint=False)
    array = np.arange(0, 100)
    indices = (wavelength > 10) & (wavelength < 50)
    new_wavelength, new_array = utils.restrict_wavelength(wavelength, array, [10, 50])
    assert new_wavelength.size == wavelength[indices].size
    assert new_array.size == array[indices].size


@pytest.mark.parametrize("shape", [(1, 10), (2, 10), (3, 10)])
def test_restrict_phase(shape: Tuple[int, int]) -> None:
    """Tests the restriction of the phase grid."""
    phases = np.random.rand(*shape) * 360
    new_phases = utils.restrict_phase(phases)
    assert ((phases > 180) | (phases < -180)).any()
    assert ((new_phases < 180) | (new_phases > -180)).all()


# def test_get_opacity() -> None:
#     """Tests the retrieval of the opacity."""
#     ...


def test_load_data(qval_files: List[Path], grf_files: List[Path]) -> None:
    """Tests the loading of a data file."""
    wavelength, data = utils.load_data(qval_files[0], load_func=utils.qval_to_opacity)
    assert len(wavelength.shape) == 1
    assert len(data.shape) == 1

    wavelength, data = utils.load_data(qval_files, load_func=utils.qval_to_opacity)
    assert len(wavelength.shape) == 1
    assert data.shape[0] == 6

    wavelength, data = utils.load_data(grf_files)
    assert len(wavelength.shape) == 1
    assert data.shape[0] == 6


def test_linearly_combine_data(
    qval_files: List[Path],
    grf_files: List[Path],
    continuum_file: Path,
    high_wavelength_solution: u.um,
    wavelength_solutions: u.um,
    all_wavelength_grid: u.um,
) -> None:
    """Tests the linear combination of interpolated wavelength grids."""
    low_wavelength_solution = wavelength_solutions[1].value
    high_wavelength_solution = high_wavelength_solution.value
    all_wavelength_grid = all_wavelength_grid.value

    opacity_dir = Path("opacities")
    if not opacity_dir.exists():
        opacity_dir.mkdir()

    weights = np.array([42.8, 9.7, 43.5, 1.1, 2.3, 0.6]) / 100
    wavelength_qval, qval = utils.load_data(qval_files, load_func=utils.qval_to_opacity)
    qval_combined = utils.linearly_combine_data(qval, weights)

    wavelength_grf, grf = utils.load_data(grf_files)
    grf_combined = utils.linearly_combine_data(grf, weights)

    assert len(grf_combined.shape) == 1
    assert len(qval_combined.shape) == 1

    wavelength_cont, continuum_data = np.load(continuum_file)

    cont_low = np.interp(low_wavelength_solution, wavelength_cont, continuum_data)
    cont_high = np.interp(high_wavelength_solution, wavelength_cont, continuum_data)
    qval_low = np.interp(low_wavelength_solution, wavelength_qval, qval_combined)
    qval_high = np.interp(high_wavelength_solution, wavelength_qval, qval_combined)
    grf_low = np.interp(low_wavelength_solution, wavelength_grf, grf_combined)
    grf_high = np.interp(high_wavelength_solution, wavelength_grf, grf_combined)

    cont_ind = (wavelength_cont >= 1) & (wavelength_cont <= low_wavelength_solution[-1])
    qval_ind = (wavelength_qval >= 1) & (wavelength_qval <= low_wavelength_solution[-1])
    grf_ind = (wavelength_grf >= 1) & (wavelength_grf <= low_wavelength_solution[-1])

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

    cont_intp = np.interp(all_wavelength_grid, wavelength_cont, continuum_data)
    qval_intp = np.interp(all_wavelength_grid, wavelength_qval, qval_combined)
    grf_intp = np.interp(all_wavelength_grid, wavelength_grf, grf_combined)
    cont_ind = (wavelength_cont >= all_wavelength_grid[0]) & (
        wavelength_cont <= all_wavelength_grid[-1]
    )
    qval_ind = (wavelength_qval >= 1) & (wavelength_qval <= all_wavelength_grid[-1])
    grf_ind = (wavelength_grf >= 1) & (wavelength_grf <= all_wavelength_grid[-1])

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


@pytest.mark.parametrize("wavelength", [[10] * u.um, [10, 12.5] * u.um])
def test_broadcast_baselines(fits_files: List[Path], wavelength: u.um) -> None:
    """Tests the broadcasting of the baselines."""
    set_data(fits_files, wavelengths=wavelength, fit_data=["vis2", "t3"])

    vis, t3 = OPTIONS.data.vis2, OPTIONS.data.t3
    baselines, baseline_angles = utils.compute_effective_baselines(
        vis.ucoord, vis.vcoord, None, None
    )
    wavelengths_vis, baselines_vis, baseline_angles_vis = utils.broadcast_baselines(
        wavelength, baselines, baseline_angles, vis.ucoord
    )

    assert wavelengths_vis.shape == (wavelength.size, 1)
    assert baselines_vis.shape == (wavelength.size, vis.ucoord.shape[1], 1)
    assert baseline_angles_vis.shape == (1, vis.ucoord.shape[1], 1)

    baselines, baseline_angles = utils.compute_effective_baselines(
        t3.u123coord, t3.v123coord, None, None
    )
    wavelengths_cp, baselines_cp, baseline_angles_cp = utils.broadcast_baselines(
        wavelength, baselines, baseline_angles, t3.u123coord
    )

    assert wavelengths_cp.shape == (wavelength.size, 1, 1)
    assert baselines_cp.shape == (wavelength.size, *t3.u123coord.shape, 1)
    assert baseline_angles_cp.shape == (1, *t3.u123coord.shape, 1)

    set_data(fit_data=["vis2", "t3"])


# TODO: This test doesn't test much
@pytest.mark.parametrize(
    "component, wavelength",
    [
        (comp, wl)
        for wl in [[10] * u.um, [10, 12.5] * u.um]
        for comp in [PointSource, Star, Gaussian, TempGradient]
    ],
)
def test_compute_t3(
    fits_files: List[Path], wavelength: u.um, component: FourierComponent
) -> None:
    """Tests the calculation of the closure phase."""
    set_data(fits_files, wavelengths=wavelength, fit_data=["t3"])
    fr = Parameter(**STANDARD_PARAMS.fr)
    fr.value, fr.grid = np.array([0.2] * wavelength.size), wavelength

    params = {
        "dim": 512,
        "fwhm": 0.5,
        "rin": 0.5,
        "q": 0.5,
        "temp0": 1500,
        "dim": 512,
        "dist": 148.3,
        "eff_temp": 7800,
        "eff_radius": 1.8,
        "sigma0": 2000,
        "pixel_size": 0.1,
        "p": 0.5,
    }

    t3 = OPTIONS.data.t3
    component = component(**params)
    if component.name == "Point":
        component.fr = fr
    component_vis = component.compute_complex_vis(
        t3.u123coord, t3.v123coord, wavelength
    )
    component_t3 = utils.compute_t3(component_vis)

    assert component_vis.shape == (wavelength.size, *t3.u123coord.shape)
    assert component_t3.shape == (wavelength.size, t3.u123coord.shape[1])

    set_data(fit_data=["t3"])


@pytest.mark.parametrize(
    "component, wavelength",
    [
        (comp, wl)
        for wl in [[10] * u.um, [10, 12.5] * u.um]
        for comp in [PointSource, Star, Gaussian, TempGradient]
    ],
)
def test_compute_vis(
    fits_files: List[Path], wavelength: u.um, component: FourierComponent
) -> None:
    """Tests the calculation of the visibility."""
    set_data(fits_files, wavelengths=wavelength, fit_data=["vis2"])
    fr = Parameter(**STANDARD_PARAMS.fr)
    fr.value, fr.grid = np.array([0.2] * wavelength.size), wavelength

    params = {
        "dim": 512,
        "fwhm": 0.5,
        "rin": 0.5,
        "q": 0.5,
        "temp0": 1500,
        "dim": 512,
        "dist": 148.3,
        "eff_temp": 7800,
        "eff_radius": 1.8,
        "sigma0": 2000,
        "pixel_size": 0.1,
        "p": 0.5,
    }

    vis = OPTIONS.data.vis2
    component = component(**params)
    if component.name == "Point":
        component.fr = fr
    component_complex_vis = component.compute_complex_vis(
        vis.ucoord, vis.vcoord, wavelength
    )
    component_vis = utils.compute_vis(component_complex_vis)

    assert component_complex_vis.shape == (wavelength.size, vis.ucoord.shape[1])
    assert component_vis.shape == (wavelength.size, vis.ucoord.shape[1])

    set_data(fit_data=["vis2"])
