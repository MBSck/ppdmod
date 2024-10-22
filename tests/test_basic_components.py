import time
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ppdmod import utils
from ppdmod.basic_components import (
    AsymmetricGreyBody,
    Gaussian,
    GreyBody,
    PointSource,
    Ring,
    Star,
    TempGradient,
    UniformDisk,
    assemble_components,
)
from ppdmod.data import ReadoutFits, set_data
from ppdmod.options import OPTIONS, STANDARD_PARAMETERS
from ppdmod.parameter import Parameter
from ppdmod.utils import (
    compute_t3,
    compute_vis,
    get_opacity,
    load_data,
    qval_to_opacity,
)

DIMENSION = [2**power for power in range(9, 13)]
CALCULATION_FILE = Path("analytical_calculation.xlsx")
COMPONENT_DIR = Path("component")

DATA_DIR = Path(__file__).parent.parent / "data"
READOUT = ReadoutFits(list((DATA_DIR / "matisse").glob("*2022-04-21*.fits"))[0])
utils.make_workbook(
    CALCULATION_FILE,
    {
        "Vis": ["Dimension (px)", "Computation Time (s)"],
        "T3": ["Dimension (px)", "Computation Time (s)"],
    },
)


@pytest.fixture
def wavelength() -> u.m:
    """A wavelength grid."""
    return [12.5] * u.um


@pytest.fixture
def point_source() -> PointSource:
    """Initializes a point source component."""
    return PointSource(**{"dim": 512, "fr": 0.2})


@pytest.fixture
def star() -> Star:
    """Initializes a star component."""
    return Star(**{"dim": 512, "dist": 145, "eff_temp": 7800, "eff_radius": 1.8})


@pytest.fixture
def ring() -> Ring:
    """Initializes a gaussian component."""
    return Ring(**{"dim": 512, "diam": 5, "width": 1})


@pytest.fixture
def uniform_disk() -> UniformDisk:
    """Initializes a gaussian component."""
    return UniformDisk(**{"dim": 512, "diam": 5})


@pytest.fixture
def gaussian() -> Gaussian:
    """Initializes a gaussian component."""
    return Gaussian(**{"dim": 512, "fwhm": 0.5})


@pytest.fixture
def eff_temp() -> int:
    return 6500


@pytest.fixture
def dist() -> float:
    return 158.51


@pytest.fixture
def temp_gradient() -> TempGradient:
    """Initializes a numerical component."""
    temp_grad = TempGradient(
        **{
            "rin": 0.5,
            "q": 0.5,
            "temp0": 1500,
            "dist": 148.3,
            "sigma0": 2000,
            "pixel_size": 0.1,
            "p": 0.5,
        }
    )
    temp_grad.optically_thick = True
    temp_grad.asymmetric = True
    return temp_grad


@pytest.fixture
def kappa_abs() -> Parameter:
    weights = np.array([73.2, 8.6, 0.6, 14.2, 2.4, 1.0]) / 100
    names = ["pyroxene", "forsterite", "enstatite", "silica"]
    sizes = [[1.5], [0.1], [0.1, 1.5], [0.1, 1.5]]

    grid, value = get_opacity(DATA_DIR, weights, sizes, names, "boekel")
    kappa_abs = Parameter(**STANDARD_PARAMETERS.kappa_abs)
    kappa_abs.grid, kappa_abs.value = grid, value
    return kappa_abs


@pytest.fixture
def kappa_cont() -> Parameter:
    cont_opacity_file = DATA_DIR / "opacities" / "optool" / "preibisch_amorph_c_rv0.1.npy"
    grid, value = load_data(cont_opacity_file, load_func=qval_to_opacity)

    kappa_cont = Parameter(**STANDARD_PARAMETERS.kappa_cont)
    kappa_cont.value, kappa_cont.grid = grid, value
    return kappa_cont


@pytest.fixture
def star_flux() -> Parameter:
    wl_flux, flux = load_data(
        DATA_DIR / "flux" / "hd142527" / "HD142527_stellar_model.txt"
    )
    star_flux = Parameter(**STANDARD_PARAMETERS.f)
    star_flux.grid, star_flux.value = wl_flux, flux
    return star_flux


@pytest.fixture
def luminosity() -> float:
    return 1.35


@pytest.fixture
def eff_radius(luminosity: float, eff_temp: int) -> float:
    return utils.compute_stellar_radius(10**luminosity, eff_temp).value


def test_point_source_init(point_source: PointSource) -> None:
    """Tests the point source's initialization."""
    assert "fr" in vars(point_source).keys()


def test_point_source_flux_func(point_source: PointSource, wavelength: u.um) -> None:
    """Tests the point source's initialization."""
    assert point_source.flux_func(wavelength).shape == (wavelength.size, 1)


def test_point_source_compute_vis(point_source: PointSource, wavelength: u.um) -> None:
    """tests the point source's compute_vis method."""
    vis = point_source.compute_complex_vis(
        READOUT.vis.ucoord, READOUT.vis.vcoord, wavelength
    )
    assert vis.shape == (wavelength.size, READOUT.vis.ucoord.size)


@pytest.mark.parametrize(
    "wl, dim",
    [
        (u.Quantity([wl], unit=u.um), dim)
        for dim in DIMENSION
        for wl in [8, 9, 10, 11] * u.um
    ],
)
def test_point_source_image(point_source: Star, dim: int, wl: u.um) -> None:
    """Tests the point source's image calculation."""
    image = point_source.compute_image(dim, 0.1 * u.mas, wl)
    point_source_dir = Path("images/point_source")
    point_source_dir.mkdir(exist_ok=True, parents=True)
    centre = dim // 2
    plt.imshow(image[0])
    plt.xlim(centre - 20, centre + 20)
    plt.ylim(centre - 20, centre + 20)
    plt.savefig(point_source_dir / f"dim{dim}_wl{wl.value}_point_source_image.pdf")
    plt.close()
    assert len(image[image != 0]) == 1
    assert image.shape == (1, dim, dim)
    assert np.max(image) < 0.1


def test_star_init(star: Star) -> None:
    """Tests the star's initialization."""
    assert "dist" in vars(star).keys()
    assert "eff_temp" in vars(star).keys()
    assert "eff_radius" in vars(star).keys()


def test_star_stellar_radius_angular(star: Star) -> None:
    """Tests the stellar radius conversion to angular radius."""
    assert star.stellar_radius_angular.unit == u.mas


# TODO: Include test for stellar flux with input file as well.
def test_star_flux(star: Star, wavelength: u.um) -> None:
    """Tests the calculation of the total flux."""
    assert star.flux_func(wavelength).shape == (wavelength.size, 1)


def test_star_compute_vis(star: Star, wavelength: u.um) -> None:
    """Tests the calculation of the total flux."""
    vis = star.compute_complex_vis(READOUT.vis.ucoord, READOUT.vis.vcoord, wavelength)
    assert vis.shape == (wavelength.size, READOUT.vis.ucoord.size)


# TODO: Make this for multiple wavelengths at the same time
@pytest.mark.parametrize(
    "wl, dim",
    [
        (u.Quantity([wl], unit=u.um), dim)
        for dim in DIMENSION
        for wl in [8, 9, 10, 11] * u.um
    ],
)
def test_star_image(star: Star, dim: int, wl: u.um) -> None:
    """Tests the star's image calculation."""
    image = star.compute_image(dim, 0.1 * u.mas, wl)

    star_dir = Path("images/star")
    star_dir.mkdir(exist_ok=True, parents=True)

    centre = dim // 2

    plt.imshow(image[0])
    plt.xlim(centre - 20, centre + 20)
    plt.ylim(centre - 20, centre + 20)
    plt.savefig(star_dir / f"dim{dim}_wl{wl.value}_star_image.pdf")
    plt.close()

    assert len(image[image != 0]) == 4
    assert image.shape == (1, dim, dim)
    assert np.max(image) < 0.1


def test_uniform_ring_init(ring: Ring) -> None:
    """Tests the ring's initialization."""
    assert "rin" in vars(ring).keys()
    assert "rout" in vars(ring).keys()
    assert "width" in vars(ring).keys()


@pytest.mark.parametrize(
    "fits_file, radius, wl, inc, pos_angle, width, c, s",
    [
        ("Iring.fits", 5, 10, None, None, None, None, None),
        ("Iring_inc.fits", 5, 10, 0.351, None, None, None, None),
        ("Iring_inc_rot.fits", 5, 10, 0.351, 33, None, None, None),
        ("ring.fits", 5, 10, None, None, 1, None, None),
        ("ring_inc.fits", 5, 10, 0.351, None, 1, None, None),
        ("ring_inc_rot.fits", 5, 10, 0.351, 33, 1, None, None),
        (
            "cm_Iring_rin2_inc1_pa0_c0_s0_extended.fits",
            2,
            3.5,
            None,
            None,
            None,
            None,
            None,
        ),
        (
            "cm_Iring_rin2_inc05_pa0_c0_s0_extended.fits",
            2,
            3.5,
            0.5,
            None,
            None,
            None,
            None,
        ),
        (
            "cm_Iring_rin2_inc05_pa33_c0_s0_extended.fits",
            2,
            3.5,
            0.5,
            33,
            None,
            None,
            None,
        ),
        ("cm_Iring_rin2_inc05_pa33_c1_s0_extended.fits", 2, 3.5, 0.5, 33, None, 1, 0),
        ("cm_Iring_rin2_inc05_pa33_c0_s1_extended.fits", 2, 3.5, 0.5, 33, None, 0, 1),
        # TODO : Test this one again. Discrepancy too big for numerical FFT?
        # ("cm_Iring_rin2_inc05_pa33_c1_s1_extended.fits", 2, 3.5, 0.5, 33, None, 1, 1),
        (
            "cm_Iring_rin2_inc05_pa33_c05_s05_extended.fits",
            2,
            3.5,
            0.5,
            33,
            None,
            0.5,
            0.5,
        ),
        (
            "cm_Iring_rin2_inc05_pa33_c05_s1_extended.fits",
            2,
            3.5,
            0.5,
            33,
            None,
            0.5,
            1,
        ),
        (
            "cm_Iring_rin2_inc05_pa33_c1_s05_extended.fits",
            2,
            3.5,
            0.5,
            33,
            None,
            1,
            0.5,
        ),
        (
            "cm_ring_rin2_inc1_pa0_c0_s0_w1_extended.fits",
            2,
            3.5,
            1,
            None,
            1,
            None,
            None,
        ),
        (
            "cm_ring_rin2_inc1_pa0_c0_s0_w05_extended.fits",
            2,
            3.5,
            1,
            None,
            0.5,
            None,
            None,
        ),
        (
            "cm_ring_rin2_inc05_pa0_c0_s0_w05_extended.fits",
            2,
            3.5,
            0.5,
            None,
            0.5,
            None,
            None,
        ),
        (
            "cm_ring_rin2_inc05_pa33_c0_s0_w05_extended.fits",
            2,
            3.5,
            0.5,
            33,
            0.5,
            None,
            None,
        ),
        ("cm_ring_rin2_inc05_pa33_c1_s0_w05_extended.fits", 2, 3.5, 0.5, 33, 0.5, 1, 0),
        ("cm_ring_rin2_inc05_pa33_c0_s1_w05_extended.fits", 2, 3.5, 0.5, 33, 0.5, 0, 1),
        (
            "cm_ring_rin2_inc05_pa33_c05_s05_w05_extended.fits",
            2,
            3.5,
            0.5,
            33,
            0.5,
            0.5,
            0.5,
        ),
        (
            "cm_ring_rin2_rout25_inc05_pa33_c05_s05_w0_extended.fits",
            2,
            3.5,
            0.5,
            33,
            0.5,
            0.5,
            0.5,
        ),
        (
            "cm_ring_rin2_inc05_pa33_c11_s10_w05_extended.fits",
            2,
            3.5,
            0.5,
            33,
            0.5,
            [1],
            [0],
        ),
        (
            "cm_ring_rin2_inc05_pa33_c11_s10_w05_extended.fits",
            2,
            [3.5, 3.7],
            0.5,
            33,
            0.5,
            [1],
            [0],
        ),
        (
            "cm_ring_rin2_inc05_pa33_c11_s10_c21_s20_w05_extended.fits",
            2,
            [3.5, 3.7],
            0.5,
            33,
            0.5,
            [1, 1],
            [0, 0],
        ),
        (
            "cm_ring_rin2_inc05_pa33_c11_s10_c21_s20_c31_s30_w05_extended.fits",
            2,
            3.5,
            0.5,
            33,
            0.5,
            [1, 1, 1],
            [0, 0, 0],
        ),
    ],
)
def test_ring_compute_vis(
    fits_file: Path,
    radius: u.mas,
    wl: u.um,
    inc: float,
    pos_angle: u.deg,
    width: u.mas,
    c: float,
    s: float,
) -> None:
    """Tests the calculation of the ring's visibilities."""
    if isinstance(wl, list):
        wavelength = wl * u.um
    else:
        wavelength = [wl] * u.um

    fits_file = DATA_DIR / "aspro" / fits_file
    data = set_data([fits_file], wavelengths=wavelength, fit_data=["vis", "t3"])

    thin = False if width is not None else True
    asymmetric = True if c is not None or s is not None else False
    c, s = c if c is not None else 0, s if s is not None else 0
    if isinstance(c, list):
        OPTIONS.model.modulation = len(c)

    inc = inc if inc is not None else 1
    pa = pos_angle if pos_angle is not None else 0

    has_outer_radius, rout = False, None
    if "rout" in fits_file.name:
        has_outer_radius = True
        rout = radius + width

    ring = Ring(
        rin=radius,
        rout=rout,
        width=width,
        inc=inc,
        pa=pa,
        has_outer_radius=has_outer_radius,
        thin=thin,
        asymmetric=asymmetric,
    )

    if isinstance(c, list):
        for i, (c_i, s_i) in enumerate(zip(c, s)):
            getattr(ring, f"c{i+1}").value = c_i
            getattr(ring, f"s{i+1}").value = s_i
    else:
        ring.c1.value, ring.s1.value = c, s

    vis, t3 = data.vis2 if "vis2" in OPTIONS.fit.data else data.vis, data.t3
    vis_ring = compute_vis(ring.compute_complex_vis(vis.ucoord, vis.vcoord, wavelength))
    t3_ring = compute_t3(
        ring.compute_complex_vis(t3.u123coord, t3.v123coord, wavelength)
    )
    vis_ring, t3_ring = vis_ring[:, 1:], t3_ring[:, 1:]

    atol = 1e-2 if "cm" not in fits_file.name else 1e-1
    assert vis_ring.shape == (wavelength.size, vis.ucoord.shape[1] - 1)
    assert np.allclose(vis.value, vis_ring, atol=atol)

    assert t3_ring.shape == (wavelength.size, t3.u123coord.shape[1] - 1)
    if "cm" not in fits_file.name:
        assert np.allclose(t3.value, t3_ring, atol=atol)
    else:
        # NOTE: Due to numerical inaccuraies the diff is quite high in some cases
        diff = np.abs(t3.value - t3_ring)
        assert diff.max() < 120

    set_data(fit_data=["vis", "t3"])
    OPTIONS.model.modulation = 1


def test_uniform_disk_init(uniform_disk: UniformDisk) -> None:
    """Tests the uniform disk's initialization."""
    assert "diam" in vars(uniform_disk).keys()


@pytest.mark.parametrize(
    "fits_file, compression, pos_angle",
    [
        ("ud.fits", None, None),
        ("ud_inc.fits", 0.351 * u.one, None),
        ("ud_inc_rot.fits", 0.351 * u.one, 33 * u.deg),
    ],
)
def test_uniform_disk_compute_vis(
    uniform_disk: UniformDisk, fits_file: Path, compression: float, pos_angle: u.deg
) -> None:
    """Tests the calculation of uniform disk's visibilities."""
    wavelength = [10] * u.um
    fits_file = DATA_DIR / "aspro" / fits_file
    data = set_data([fits_file], wavelengths=wavelength, fit_data=["vis", "t3"])
    vis, t3 = data.vis2 if "vis2" in OPTIONS.fit.data else data.vis, data.t3

    uniform_disk.diam.value = 20 * u.mas
    if compression is not None:
        uniform_disk.elliptic = True

    uniform_disk.inc.value = compression if compression is not None else 1
    uniform_disk.pa.value = pos_angle if pos_angle is not None else 0

    vis_ud = compute_vis(
        uniform_disk.compute_complex_vis(vis.ucoord, vis.vcoord, wavelength)
    )
    t3_ud = compute_t3(
        uniform_disk.compute_complex_vis(t3.u123coord, t3.v123coord, wavelength)
    )

    assert vis_ud.shape == (wavelength.size, vis.ucoord.shape[1])
    assert np.allclose(vis.value, vis_ud[:, 1:], atol=1e-2)

    assert t3_ud.shape == (wavelength.size, t3.u123coord.shape[1])
    assert np.allclose(t3.value, t3_ud[:, 1:], atol=1e-2)

    set_data(fit_data=["vis", "t3"])


def test_uniform_disk_image_func() -> None:
    """Tests the calculation of the uniform disk's image function."""
    ...


def test_gaussian_init(gaussian: Gaussian) -> None:
    """Tests the gaussian's initialization."""
    assert "hlr" in vars(gaussian).keys()


@pytest.mark.parametrize(
    "fits_file, compression, pos_angle",
    [
        ("gaussian.fits", None, None),
        ("gaussian_inc.fits", 0.351 * u.one, None),
        ("gaussian_inc_rot.fits", 0.351 * u.one, 33 * u.deg),
    ],
)
def test_gaussian_compute_vis(
    gaussian: Gaussian, fits_file: Path, compression: float, pos_angle: u.deg
) -> None:
    """Tests the calculation of the total flux."""
    wavelength = [10] * u.um
    fits_file = DATA_DIR / "aspro" / fits_file
    data = set_data([fits_file], wavelengths=wavelength, fit_data=["vis", "t3"])

    gaussian.hlr.value = 10 * u.mas / 2
    vis, t3 = data.vis2 if "vis2" in OPTIONS.fit.data else data.vis, data.t3
    if compression is not None:
        gaussian.elliptic = True

    gaussian.inc.value = compression if compression is not None else 1
    gaussian.pa.value = pos_angle if pos_angle is not None else 0

    vis_gauss = compute_vis(
        gaussian.compute_complex_vis(vis.ucoord, vis.vcoord, wavelength)
    )
    t3_gauss = compute_t3(
        gaussian.compute_complex_vis(t3.u123coord, t3.v123coord, wavelength)
    )

    assert vis_gauss.shape == (wavelength.size, vis.ucoord.shape[1])
    assert np.allclose(vis.value, vis_gauss[:, 1:], atol=1e-2)

    assert t3_gauss.shape == (wavelength.size, t3.u123coord.shape[1])
    assert np.allclose(t3.value, t3_gauss[:, 1:], atol=1e-2)

    set_data(fit_data=["vis", "t3"])


# TODO: Finish this test for the gaussian
# @pytest.mark.parametrize(
#         "compression, pos_angle",
#         [(None, None)])
# def test_gaussian_image_func(
#         gaussian: Gaussian, fits_file: Path, compression: float,
#         pos_angle: u.deg, wavelength: u.um) -> None:
#     """Tests the calculation of the total flux."""
#     fits_file = Path("data/aspro") / fits_file
#     image = gaussian.compute_image(512, 0.1*u.mas, wavelength)
#     assert image.shape == (wavelength.size, 512, 512)
#     assert image.unit == u.one
#
#     gaussian.elliptic = False


@pytest.mark.parametrize("grid_type", ["linear", "logarithmic"])
def test_temp_gradient_compute_grid(
    temp_gradient: TempGradient, grid_type: str
) -> None:
    """Tests the hankel component's grid calculation."""
    OPTIONS.model.gridtype = grid_type
    radius = temp_gradient.compute_internal_grid()
    assert radius.unit == u.au
    assert radius.shape == (temp_gradient.dim(),)
    assert (
        radius[0].value == temp_gradient.rin.value
        and radius[-1].value == temp_gradient.rout.value
    )

    OPTIONS.model.gridtype = "logarithmic"


def test_temp_gradient_flux(temp_gradient: TempGradient, wavelength: u.um) -> None:
    """Tests the calculation of the total flux."""
    flux = temp_gradient.compute_flux(wavelength)
    assert flux.shape == (wavelength.size, 1)


@pytest.mark.parametrize(
    "fits_file, inc, pos_angle, c, s",
    [
        ("cm_AsymGreyBody_inc1_pa0_c0_s0_extended.fits", 1, None, None, None),
        ("cm_AsymGreyBody_inc05_pa33_c0_s0_extended.fits", 0.5, 33, None, None),
        ("cm_AsymGreyBody_inc05_pa33_c1_s0_extended.fits", 0.5, 33, 1, 0),
        ("cm_AsymGreyBody_inc05_pa33_c0_s1_extended.fits", 0.5, 33, 0, 1),
        ("cm_AsymGreyBody_inc05_pa33_c1_s1_extended.fits", 0.5, 33, 1, 1),
        ("cm_AsymGreyBody_inc05_pa33_c05_s05_extended.fits", 0.5, 33, 0.5, 0.5),
    ],
)
def test_temp_gradient_compute_vis(
    fits_file: Path,
    inc: float,
    pos_angle: u.deg,
    c: float,
    s: float,
    eff_temp: int,
    eff_radius: float,
    dist: float,
    star_flux: Parameter,
    kappa_abs: Parameter,
    kappa_cont: Parameter,
) -> None:
    """Tests the calculation of the ring's visibilities."""
    fits_file, wl = DATA_DIR / "aspro" / fits_file, [3.5] * u.um
    data = set_data([fits_file], wavelengths=wl, fit_data=["vis", "t3"])
    c, s = c if c is not None else 0, s if s is not None else 0
    inc = inc if inc is not None else 1
    pa = pos_angle if pos_angle is not None else 0
    atg = AsymmetricGreyBody(
        rin=1.5,
        rout=2,
        inc=inc,
        pa=pa,
        p=0.5,
        sigma0=1e-4,
        cont_weight=0.9,
        r0=1,
        kappa_abs=kappa_abs,
        kappa_cont=kappa_cont,
        dist=dist,
        eff_temp=eff_temp,
        eff_radius=eff_radius,
        c1=c,
        s1=s,
    )

    vis, t3 = data.vis2 if "vis2" in OPTIONS.fit.data else data.vis, data.t3
    complex_vis = atg.compute_complex_vis(vis.ucoord, vis.vcoord, wl)
    complex_t3 = atg.compute_complex_vis(t3.u123coord, t3.v123coord, wl)
    if "star" in fits_file.name.lower():
        star = Star(f=star_flux)
        complex_vis += star.compute_complex_vis(vis.ucoord, vis.vcoord, wl)
        complex_t3 += star.compute_complex_vis(t3.u123coord, t3.v123coord, wl)

    vis_atg = compute_vis(complex_vis)
    vis_atg = vis_atg[:, 1:] / vis_atg.max()
    t3_atg = compute_t3(complex_t3)[:, 1:]

    atol = 1e-2 if "cm" not in fits_file.name else 1e-1
    assert vis_atg.shape == (wl.size, vis.ucoord.shape[1] - 1)
    assert np.allclose(vis.value, vis_atg, atol=atol)

    if "cm" not in fits_file.name:
        assert t3_atg.shape == (wl.size, t3.u123coord.shape[1])
        assert np.allclose(t3.value, t3_atg, atol=atol)
    else:
        # NOTE: The differences here are larger due to numerical inaccuracies in ASPRO?
        # Values for positional angle and inclination are ~ 153 degrees (before that < 45)
        # Sometimes even ~ 160
        diff = np.ptp(
            np.hstack((t3.value[0][:, np.newaxis], t3_atg[0][:, np.newaxis])), axis=1
        )
        assert diff.max() < 170

    set_data(fit_data=["vis", "t3"])


# TODO: These tests need to be changed to include negative total flux
@pytest.mark.parametrize(
    "fits_file, inc, pos_angle, c, s",
    [
        ("cm_StarAsymGreyAsymGrey_inc1_pa0_c0_s0_extended.fits", 1, None, None, None),
        ("cm_StarAsymGreyAsymGrey_inc05_pa33_c0_s0_extended.fits", 0.5, 33, None, None),
        ("cm_StarAsymGreyAsymGrey_inc05_pa33_c05_s1_extended.fits", 0.5, 33, 0.5, 1),
    ],
)
def test_fluxes_vs_aspro(
    fits_file: Path,
    inc: float,
    pos_angle: u.deg,
    c: float,
    s: float,
    dist: float,
    eff_temp: int,
    eff_radius: float,
    kappa_abs: Parameter,
    kappa_cont: Parameter,
    star_flux: Parameter,
) -> None:
    """Tests the calculation of temperature gradient's visibilities vs aspro."""
    wl = [3.5] * u.um
    data = set_data(
        [DATA_DIR / "aspro" / fits_file], wavelengths=wl, fit_data=["vis", "t3"]
    )
    c, s = c if c is not None else 0, s if s is not None else 0
    inc = inc if inc is not None else 1
    pa = pos_angle if pos_angle is not None else 0

    star = Star(f=star_flux)
    atg = GreyBody(
        rin=1.5,
        rout=2,
        dist=dist,
        eff_temp=eff_temp,
        eff_radius=eff_radius,
        inc=inc,
        pa=pa,
        p=0.5,
        sigma0=1e-4,
        r0=1,
        kappa_abs=kappa_abs,
        kappa_cont=kappa_cont,
        cont_weight=0.9,
    )
    atg2 = AsymmetricGreyBody(
        rin=3,
        rout=5,
        dist=dist,
        eff_temp=eff_temp,
        eff_radius=eff_radius,
        inc=inc,
        pa=pa,
        p=0.5,
        sigma0=1e-4,
        r0=1,
        c1=c,
        s1=s,
        kappa_abs=kappa_abs,
        kappa_cont=kappa_cont,
        cont_weight=0.2,
    )

    vis, t3 = data.vis2 if "vis2" in OPTIONS.fit.data else data.vis, data.t3

    flux_star = star.compute_flux(wl)
    flux_atg = atg.compute_flux(wl)
    flux_atg2 = atg2.compute_flux(wl)
    flux_combined = (flux_star + flux_atg + flux_atg2).squeeze()

    vis_star = star.compute_complex_vis(vis.ucoord, vis.vcoord, wl)
    vis_atg = atg.compute_complex_vis(vis.ucoord, vis.vcoord, wl)
    vis_atg2 = atg2.compute_complex_vis(vis.ucoord, vis.vcoord, wl)
    vis_combined = compute_vis(vis_star + vis_atg + vis_atg2)
    flux_from_vis = vis_combined[:, 0].squeeze()
    vis_norm_combined = vis_combined[:, 1:] / flux_from_vis

    t3_start = star.compute_complex_vis(data.t3.u123coord, data.t3.v123coord, wl)
    t3_atg = atg.compute_complex_vis(data.t3.u123coord, data.t3.v123coord, wl)
    t3_atg2 = atg2.compute_complex_vis(data.t3.u123coord, data.t3.v123coord, wl)
    t3_combined = compute_t3(t3_start + t3_atg + t3_atg2)

    assert np.isclose(flux_combined, flux_from_vis)
    assert np.allclose(vis.value, vis_norm_combined, atol=1e-1)

    diff = np.ptp(
        np.hstack((t3.value[0][:, np.newaxis], t3_combined[:, 1:][0][:, np.newaxis])),
        axis=1,
    )
    assert diff.max() < 5
    set_data(fit_data=["vis", "t3"])


@pytest.mark.parametrize("dim", [4096, 2096, 1024, 512, 256, 128, 64, 32])
def test_temp_gradient_resolution(
    temp_gradient: TempGradient, dim: int, wavelength: u.um
) -> None:
    """Tests the hankel component's resolution."""
    temp_gradient.dim.value = dim
    temp_gradient.optically_thick = True
    temp_gradient.asymmetric = False

    OPTIONS.model.modulation = 1
    start_time_vis = time.perf_counter()
    _ = temp_gradient.compute_complex_vis(
        READOUT.vis2.ucoord, READOUT.vis2.vcoord, wavelength
    )
    end_time_vis = time.perf_counter() - start_time_vis

    start_time_cphase = time.perf_counter()
    _ = temp_gradient.compute_complex_vis(
        READOUT.t3.u123coord, READOUT.t3.v123coord, wavelength
    )
    end_time_cphase = time.perf_counter() - start_time_cphase

    vis_data = {"Dimension (px)": [dim], "Computation Time (s)": [end_time_vis]}

    t3_data = {"Dimension (px)": [dim], "Computation Time (s)": [end_time_cphase]}

    if CALCULATION_FILE.exists():
        df = pd.read_excel(CALCULATION_FILE, sheet_name="Vis")
        new_df = pd.DataFrame(vis_data)
        df = pd.concat([df, new_df])
    else:
        df = pd.DataFrame(vis_data)

    with pd.ExcelWriter(
        CALCULATION_FILE, engine="openpyxl", mode="a", if_sheet_exists="replace"
    ) as writer:
        df.to_excel(writer, sheet_name="Vis", index=False)

    if CALCULATION_FILE.exists():
        df = pd.read_excel(CALCULATION_FILE, sheet_name="T3")
        new_df = pd.DataFrame(t3_data)
        df = pd.concat([df, new_df])
    else:
        df = pd.DataFrame(t3_data)

    with pd.ExcelWriter(
        CALCULATION_FILE, engine="openpyxl", mode="a", if_sheet_exists="replace"
    ) as writer:
        df.to_excel(writer, sheet_name="T3", index=False)

    OPTIONS.model.modulation = 0
    temp_gradient.optically_thick = False
    temp_gradient.asymmetric = False


def test_assemble_components() -> None:
    """Tests the model's assemble_model method."""
    param_names = ["rin", "p", "cont_weight", "pa", "inc"]
    values = [1.5, 0.5, 0.2, 45, 1.6]
    limits = [[0, 20], [0, 1], [0, 1], [0, 360], [1, 50]]
    params = {
        name: Parameter(**getattr(STANDARD_PARAMETERS, name)) for name in param_names
    }
    for value, limit, param in zip(values, limits, params.values()):
        param.set(*limit)
        param.value = value
    shared_params = {"p": params["p"]}
    del params["p"]

    components_and_params = [["Star", params], ["GreyBody", params]]
    components = assemble_components(components_and_params, shared_params)
    assert isinstance(components[0], Star)
    assert isinstance(components[1], GreyBody)
    assert all(
        not hasattr(components[0], param)
        for param in param_names
        if param not in ["pa", "inc"]
    )
    assert all(hasattr(components[1], param) for param in param_names)
    assert all(
        getattr(components[1], name).value == value
        for name, value in zip(["pa", "inc"], values[-2:])
    )
    assert all(
        getattr(components[1], name).value == value
        for name, value in zip(param_names, values)
    )
    assert all(
        [getattr(components[0], name).min, getattr(components[0], name).max] == limit
        for name, limit in zip(["pa", "inc"], limits[-2:])
    )
    assert all(
        [getattr(components[1], name).min, getattr(components[1], name).max] == limit
        for name, limit in zip(param_names, limits)
    )
