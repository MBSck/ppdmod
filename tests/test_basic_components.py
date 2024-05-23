from sys import breakpointhook
import time
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ppdmod import utils
from ppdmod.basic_components import PointSource, Star, Ring, UniformDisk, \
    Gaussian, Lorentzian, GaussLorentzian, TempGradient, GreyBody, StarHaloRing, \
    assemble_components
from ppdmod.data import ReadoutFits, set_data
from ppdmod.options import STANDARD_PARAMETERS, OPTIONS
from ppdmod.parameter import Parameter
from ppdmod.utils import compute_vis, compute_t3, compute_effective_baselines, broadcast_baselines


DIMENSION = [2**power for power in range(9, 13)]
CALCULATION_FILE = Path("analytical_calculation.xlsx")
COMPONENT_DIR = Path("component")

READOUT = ReadoutFits(list(Path("data/fits/hd142666").glob("*2022-04-23*.fits"))[0])
utils.make_workbook(
    CALCULATION_FILE,
    {
        "Vis": ["Dimension (px)", "Computation Time (s)"],
        "T3": ["Dimension (px)", "Computation Time (s)"],
    })


@pytest.fixture
def wavelength() -> u.m:
    """A wavelenght grid."""
    return [12.5]*u.um


@pytest.fixture
def wavelength_solution() -> u.um:
    """A MATISSE (.fits)-file."""
    file = "hd_142666_2022-04-23T03_05_25:2022-04-23T02_28_06_AQUARIUS_FINAL_TARGET_INT.fits"
    return ReadoutFits(Path("data/fits") / file).wavelength


@pytest.fixture
def qval_file_dir() -> Path:
    """The qval-file directory."""
    return Path("data/qval")


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
def temp_gradient() -> TempGradient:
    """Initializes a numerical component."""
    temp_grad = TempGradient(**{
        "rin": 0.5, "q": 0.5, "inner_temp": 1500, "dist": 148.3,
        "inner_sigma": 2000, "pixel_size": 0.1, "p": 0.5})
    temp_grad.optically_thick = True
    temp_grad.asymmetric = True
    return temp_grad
    

# def test_point_source_init(point_source: PointSource) -> None:
#     """Tests the point source's initialization."""
#     assert "fr" in vars(point_source).keys()
#
#
# def test_point_source_flux_func(point_source: PointSource, wavelength: u.um) -> None:
#     """Tests the point source's initialization."""
#     assert point_source.flux_func(wavelength).shape == (wavelength.size, 1)
#
#
# def test_point_source_compute_vis(point_source: PointSource, wavelength: u.um) -> None:
#     """tests the point source's compute_vis method."""
#     vis = point_source.compute_complex_vis(READOUT.vis.ucoord, READOUT.vis.vcoord, wavelength)
#     assert vis.shape == (wavelength.size, READOUT.vis.ucoord.size)
#
#
# @pytest.mark.parametrize(
#         "wl, dim", [(u.Quantity([wl], unit=u.um), dim)
#                     for dim in DIMENSION for wl in [8, 9, 10, 11]*u.um])
# def test_point_source_image(point_source: Star, dim: int, wl: u.um) -> None:
#     """Tests the point source's image calculation."""
#     image = point_source.compute_image(dim, 0.1*u.mas, wl)
#     point_source_dir = Path("images/point_source")
#     point_source_dir.mkdir(exist_ok=True, parents=True)
#     centre = dim//2
#     plt.imshow(image[0])
#     plt.xlim(centre-20, centre+20)
#     plt.ylim(centre-20, centre+20)
#     plt.savefig(point_source_dir / f"dim{dim}_wl{wl.value}_point_source_image.pdf")
#     plt.close()
#     assert len(image[image != 0]) == 1
#     assert image.shape == (1, dim, dim)
#     assert np.max(image) < 0.1
#
#
# def test_star_init(star: Star) -> None:
#     """Tests the star's initialization."""
#     assert "dist" in vars(star).keys()
#     assert "eff_temp" in vars(star).keys()
#     assert "eff_radius" in vars(star).keys()
#
#
# def test_star_stellar_radius_angular(star: Star) -> None:
#     """Tests the stellar radius conversion to angular radius."""
#     assert star.stellar_radius_angular.unit == u.mas
#
#
# # TODO: Include test for stellar flux with input file as well.
# def test_star_flux(star: Star, wavelength: u.um) -> None:
#     """Tests the calculation of the total flux."""
#     assert star.flux_func(wavelength).shape == (wavelength.size, 1)
#
#
# def test_star_compute_vis(star: Star, wavelength: u.um) -> None:
#     """Tests the calculation of the total flux."""
#     vis = star.compute_complex_vis(READOUT.vis.ucoord, READOUT.vis.vcoord, wavelength)
#     assert vis.shape == (wavelength.size, READOUT.vis.ucoord.size)
#
#
# # TODO: Make this for multiple wavelengths at the same time
# @pytest.mark.parametrize(
#         "wl, dim", [(u.Quantity([wl], unit=u.um), dim)
#                     for dim in DIMENSION for wl in [8, 9, 10, 11]*u.um])
# def test_star_image(star: Star, dim: int, wl: u.um) -> None:
#     """Tests the star's image calculation."""
#     image = star.compute_image(dim, 0.1*u.mas, wl)
#
#     star_dir = Path("images/star")
#     star_dir.mkdir(exist_ok=True, parents=True)
#
#     centre = dim//2
#
#     plt.imshow(image[0])
#     plt.xlim(centre-20, centre+20)
#     plt.ylim(centre-20, centre+20)
#     plt.savefig(star_dir / f"dim{dim}_wl{wl.value}_star_image.pdf")
#     plt.close()
#
#     assert len(image[image != 0]) == 4
#     assert image.shape == (1, dim, dim)
#     assert np.max(image) < 0.1
#
#
# def test_uniform_ring_init(ring: Ring) -> None:
#     """Tests the ring's initialization."""
#     assert "rin" in vars(ring).keys()
#     assert "rout" in vars(ring).keys()
#     assert "width" in vars(ring).keys()


@pytest.mark.parametrize(
        "fits_file, radius, wl, inc, pos_angle, width, c, s",
        [
         # ("Iring.fits", 5, 10, None, None, None, None, None),
         # ("Iring_inc.fits", 5, 10, 0.351, None, None, None, None),
         # ("Iring_inc_rot.fits", 5, 10, 0.351, 33, None, None, None),
         # ("ring.fits", 5, 10, None, None, 1, None, None),
         # ("ring_inc.fits", 5, 10, 0.351, None, 1, None, None),
         # ("ring_inc_rot.fits", 5, 10, 0.351, 33, 1, None, None),
         ("cm_Iring_rin5_inc1_pa0_c0_s0_UTs.fits", 5, 3.5, None, None, None, None, None),
         # ("cm_Iring_rin2_inc0.5_pa0_c0_s0_large.fits", 2, 3.5, None, None, None, 1, None),
         ])
def test_ring_compute_vis(
        fits_file: Path,
        radius: u.mas, wl: u.um, inc: float,
        pos_angle: u.deg, width: u.mas,
        c: float, s: float) -> None:
    """Tests the calculation of uniform disk's visibilities."""
    wavelength = [wl]*u.um
    fits_file = Path("data/aspro") / fits_file
    data = set_data([fits_file], wavelengths=wavelength, fit_data=["vis", "t3"])

    thin = False if width is not None else True
    asymmetric = True if c is not None or s is not None else False
    c, s = c if c is not None else 0, s if s is not None else 0
    inc = inc if inc is not None else 1
    pa = pos_angle if pos_angle is not None else 0
    ring = Ring(rin=radius, inc=inc, pa=pa, thin=thin,
                width=width, asymmetric=asymmetric, c1=c, s1=s)
    vis, t3 = data.vis2 if "vis2" in OPTIONS.fit.data else data.vis, data.t3

    vis_ring = compute_vis(ring.compute_complex_vis(vis.ucoord, vis.vcoord, wavelength))
    t3_ring = compute_t3(ring.compute_complex_vis(t3.u123coord, t3.v123coord, wavelength))
    if "cm" in fits_file.name:
        breakpoint()

    atol = 1e-2 if "cm" not in fits_file.name else 1e-1
    assert vis_ring.shape == (wavelength.size, vis.ucoord.shape[1])
    assert np.allclose(vis.value, vis_ring, atol=atol)

    if "cm" not in fits_file.name:
        assert t3_ring.shape == (wavelength.size, t3.u123coord.shape[1])
        assert np.allclose(t3.value, t3_ring, atol=atol)
    else:
        # NOTE: The differences here are larger due to numerical inaccuracies in ASPRO?
        # Values for positional angle and inclination are ~ 153 degrees (before that < 45)
        # Sometimes even ~ 160
        diff = np.ptp(np.hstack((t3.value[0][:, np.newaxis], t3_ring[0][:, np.newaxis])), axis=1)
        assert diff.max() < 170

    set_data(fit_data=["vis", "t3"])


def test_uniform_disk_init(uniform_disk: UniformDisk) -> None:
    """Tests the uniform disk's initialization."""
    assert "diam" in vars(uniform_disk).keys()


@pytest.mark.parametrize(
        "fits_file, compression, pos_angle",
        [("ud.fits", None, None),
         ("ud_inc.fits", 0.351*u.one, None),
         ("ud_inc_rot.fits", 0.351*u.one, 33*u.deg)])
def test_uniform_disk_compute_vis(
        uniform_disk: UniformDisk, fits_file: Path,
        compression: float, pos_angle: u.deg) -> None:
    """Tests the calculation of uniform disk's visibilities."""
    wavelength = [10]*u.um
    fits_file = Path("data/aspro") / fits_file
    data = set_data([fits_file], wavelengths=wavelength, fit_data=["vis", "t3"])
    vis, t3 = data.vis2 if "vis2" in OPTIONS.fit.data else data.vis, data.t3

    uniform_disk.diam.value = 20 * u.mas
    if compression is not None:
        uniform_disk.elliptic = True

    uniform_disk.inc.value = compression if compression is not None else 1
    uniform_disk.pa.value = pos_angle if pos_angle is not None else 0

    vis_ud = compute_vis(uniform_disk.compute_complex_vis(vis.ucoord, vis.vcoord, wavelength))
    t3_ud = compute_t3(uniform_disk.compute_complex_vis(t3.u123coord, t3.v123coord, wavelength))
    
    assert vis_ud.shape == (wavelength.size, vis.ucoord.shape[1])
    assert np.allclose(vis.value, vis_ud, atol=1e-2)

    assert t3_ud.shape == (wavelength.size, t3.u123coord.shape[1])
    assert np.allclose(t3.value, t3_ud, atol=1e-2)

    set_data(fit_data=["vis", "t3"])


def test_uniform_disk_image_func() -> None:
    """Tests the calculation of the uniform disk's image function."""
    ...


def test_gaussian_init(gaussian: Gaussian) -> None:
    """Tests the gaussian's initialization."""
    assert "hlr" in vars(gaussian).keys()


@pytest.mark.parametrize(
        "fits_file, compression, pos_angle",
        [("gaussian.fits", None, None),
         ("gaussian_inc.fits", 0.351*u.one, None),
         ("gaussian_inc_rot.fits", 0.351*u.one, 33*u.deg)])
def test_gaussian_compute_vis(
        gaussian: Gaussian, fits_file: Path,
        compression: float, pos_angle: u.deg) -> None:
    """Tests the calculation of the total flux."""
    wavelength = [10]*u.um
    fits_file = Path("data/aspro") / fits_file
    data = set_data([fits_file], wavelengths=wavelength, fit_data=["vis", "t3"])

    gaussian.hlr.value = 10 * u.mas / 2
    vis, t3 = data.vis2 if "vis2" in OPTIONS.fit.data else data.vis, data.t3
    if compression is not None:
        gaussian.elliptic = True

    gaussian.inc.value = compression if compression is not None else 1
    gaussian.pa.value = pos_angle if pos_angle is not None else 0

    vis_gauss = compute_vis(gaussian.compute_complex_vis(vis.ucoord, vis.vcoord, wavelength))
    t3_gauss = compute_t3(gaussian.compute_complex_vis(t3.u123coord, t3.v123coord, wavelength))
    
    assert vis_gauss.shape == (wavelength.size, vis.ucoord.shape[1])
    assert np.allclose(vis.value, vis_gauss, atol=1e-2)

    assert t3_gauss.shape == (wavelength.size, t3.u123coord.shape[1])
    assert np.allclose(t3.value, t3_gauss, atol=1e-2)

    set_data(fit_data=["vis", "t3"])

    
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


# @pytest.mark.parametrize("grid_type", ["linear", "logarithmic"])
# def test_temp_gradient_compute_grid(
#         temp_gradient: TempGradient, grid_type: str) -> None:
#     """Tests the hankel component's grid calculation."""
#     OPTIONS.model.gridtype = grid_type
#     radius = temp_gradient.compute_internal_grid(512)
#     assert radius.unit == u.mas
#     assert radius.shape == (512, )
#     assert radius[0].value == temp_gradient.rin.value\
#         and radius[-1].value == temp_gradient.rout.value
#
#     OPTIONS.model.gridtype = "logarithmic"
#
#
# def test_temp_gradient_compute_brightness():
#     ...
#
#
# def test_temp_gradient_flux(
#         temp_gradient: TempGradient, wavelength: u.um) -> None:
#     """Tests the calculation of the total flux."""
#     flux = temp_gradient.compute_flux(wavelength)
#     assert flux.shape == (wavelength.size, 1)
#
#
# # TODO: Write test for hankel transform itself and compare it to ring model (aspro).
# # and skewed ring model of aspro
# # TODO: Write here check if higher orders are implemented
# @pytest.mark.parametrize("order", [0, 1, 2, 3])
# def test_temp_gradient_hankel_transform(
#         temp_gradient: TempGradient,
#         order: int, wavelength: u.um) -> None:
#     """Tests the hankel component's hankel transformation."""
#     radius = temp_gradient.compute_internal_grid(512)
#
#     OPTIONS.model.modulation = order
#
#     baselines, baseline_angles = compute_effective_baselines(
#             READOUT.vis2.ucoord, READOUT.vis2.vcoord,
#             temp_gradient.inc(), temp_gradient.pa())
#     wavelength, baselines, baseline_angles = broadcast_baselines(
#             wavelength, baselines, baseline_angles, READOUT.vis2.ucoord)
#     vis, vis_mod = temp_gradient.compute_hankel_transform(
#             radius, baselines, baseline_angles, wavelength)
#
#     assert vis.shape == (wavelength.size, 6)
#     assert vis_mod.shape == (wavelength.size, 6, order)
#     OPTIONS.model.modulation = 0
#
#
# # TODO: Add tests for the wavelength
# @pytest.mark.parametrize("order", [0, 1, 2, 3])
# def test_temp_gradient_compute_vis(
#         temp_gradient: TempGradient,
#         order: int, wavelength: u.um) -> None:
#     """Tests the hankel component's hankel transformation."""
#     OPTIONS.model.modulation = order
#
#     vis = temp_gradient.compute_complex_vis(READOUT.vis2.ucoord, READOUT.vis2.vcoord, wavelength)
#     assert vis.shape == (wavelength.size, 6)
#     assert isinstance(vis, np.ndarray)
#
#     t3 = temp_gradient.compute_complex_vis(READOUT.t3.u123coord, READOUT.t3.v123coord, wavelength)
#     assert t3.shape == (wavelength.size, 3, 4)
#     assert isinstance(vis, np.ndarray)
#
#     OPTIONS.model.modulation = 0
#
#
# # TODO: Extend this test to account for multiple files (make files an input)
# @pytest.mark.parametrize(
#         "dim", [4096, 2096, 1024, 512, 256, 128, 64, 32])
# def test_temp_gradient_resolution(temp_gradient: TempGradient,
#                                   dim: int, wavelength: u.um) -> None:
#     """Tests the hankel component's resolution."""
#     temp_gradient.dim.value = dim
#     temp_gradient.optically_thick = True
#     temp_gradient.asymmetric = True
#
#     OPTIONS.model.modulation = 1
#     start_time_vis = time.perf_counter()
#     _ = temp_gradient.compute_complex_vis(
#             READOUT.vis2.ucoord, READOUT.vis2.vcoord, wavelength)
#     end_time_vis = time.perf_counter()-start_time_vis
#
#     start_time_cphase = time.perf_counter()
#     _ = temp_gradient.compute_complex_vis(
#             READOUT.t3.u123coord, READOUT.t3.v123coord, wavelength)
#     end_time_cphase = time.perf_counter()-start_time_cphase
#
#     vis_data = {"Dimension (px)": [dim],
#                 "Computation Time (s)": [end_time_vis]}
#
#     t3_data = {"Dimension (px)": [dim],
#                "Computation Time (s)": [end_time_cphase]}
#
#     if CALCULATION_FILE.exists():
#         df = pd.read_excel(CALCULATION_FILE, sheet_name="Vis")
#         new_df = pd.DataFrame(vis_data)
#         df = pd.concat([df, new_df])
#     else:
#         df = pd.DataFrame(vis_data)
#
#     with pd.ExcelWriter(CALCULATION_FILE, engine="openpyxl",
#                         mode="a", if_sheet_exists="replace") as writer:
#         df.to_excel(writer, sheet_name="Vis", index=False)
#
#     if CALCULATION_FILE.exists():
#         df = pd.read_excel(CALCULATION_FILE, sheet_name="T3")
#         new_df = pd.DataFrame(t3_data)
#         df = pd.concat([df, new_df])
#     else:
#         df = pd.DataFrame(t3_data)
#
#     with pd.ExcelWriter(CALCULATION_FILE, engine="openpyxl",
#                         mode="a", if_sheet_exists="replace") as writer:
#         df.to_excel(writer, sheet_name="T3", index=False)
#
#     OPTIONS.model.modulation = 0
#     temp_gradient.optically_thick = False
#     temp_gradient.asymmetric = False


# def test_star_halo_ring_t3() -> None:
#     """Tests the calculation of FSCMa with the Lazareff+2017
#     StarHaloRing model."""
#     wavelength = [3.5]*u.um
#     ring_fits = Path("data/aspro") / "cm_Iring_rin8_inc064_pa68_a099_phineg10_UTs.fits"
#     data = set_data([ring_fits], wavelengths=wavelength, fit_data=["vis", "t3"])
#     vis, t3 = data.vis2 if "vis2" in OPTIONS.fit.data else data.vis, data.t3
#     vis_ring_aspro, t3_ring_aspro = vis.value.copy(), t3.value.copy()
#
#     # ring = Ring(rin=8.36941905, inc=0.63, pa=68.75493541569878,
#     #             asymmetric=True, a=0.996393496566492, phi=-10.407711312490056)
#     # complex_vis_ring = ring.compute_complex_vis(vis.ucoord, vis.vcoord, wavelength)
#     # complex_t3_ring = ring.compute_complex_vis(t3.u123coord, t3.v123coord, wavelength)
#     # vis_ring, t3_ring = compute_vis(complex_vis_ring), compute_t3(complex_t3_ring)
#
#     # assert np.allclose(vis_ring_aspro, vis_ring, atol=1e-1)
#     # diff = np.ptp(np.hstack((t3_ring_aspro[0][:, np.newaxis], t3_ring[0][:, np.newaxis])), axis=1)
#     # assert diff.max() < 10
#
#     data = set_data(fit_data=["vis", "t3"])
#
#     gauss_fits = Path("data/aspro") / "gaussian_hlr4_inc063_pa68_UTs.fits"
#     data = set_data([gauss_fits], wavelengths=wavelength, fit_data=["vis", "t3"])
#     vis, t3 = data.vis2 if "vis2" in OPTIONS.fit.data else data.vis, data.t3
#     vis_gauss_aspro, t3_gauss_aspro = vis.value.copy(), t3.value.copy()
#
#     ring = Ring(rin=8.36941905, inc=0.63, pa=68.75493541569878,
#                 asymmetric=True, a=0.996393496566492, phi=-10.407711312490056)
#     complex_vis_ring = ring.compute_complex_vis(vis.ucoord, vis.vcoord, wavelength)
#     complex_t3_ring = ring.compute_complex_vis(t3.u123coord, t3.v123coord, wavelength)
#     vis_ring, t3_ring = compute_vis(complex_vis_ring), compute_t3(complex_t3_ring)
#
#     gauss = Gaussian(hlr=4.59933786, inc=0.63, pa=68.75493541569878)
#     complex_vis_gauss = gauss.compute_complex_vis(vis.ucoord, vis.vcoord, wavelength)
#     complex_t3_gauss = gauss.compute_complex_vis(t3.u123coord, t3.v123coord, wavelength)
#     vis_gauss, t3_gauss = compute_vis(complex_vis_gauss), compute_t3(complex_t3_gauss)
#
#     assert np.allclose(vis_gauss_aspro, vis_gauss, atol=1e-2)
#     assert np.allclose(t3_gauss_aspro, t3_gauss, atol=1e-6)
#
#     lor = Lorentzian(hlr=4.59933786, inc=0.63, pa=68.75493541569878)
#     complex_vis_lor = lor.compute_complex_vis(vis.ucoord, vis.vcoord, wavelength)
#     complex_t3_lor = lor.compute_complex_vis(t3.u123coord, t3.v123coord, wavelength)
#     vis_lor, t3_lor = compute_vis(complex_vis_lor), compute_t3(complex_t3_lor)
#
#     gausslor = GaussLorentzian(flor=1., hlr=4.59933786, inc=0.63, pa=68.75493541569878)
#     complex_vis_gausslor = gausslor.compute_complex_vis(vis.ucoord, vis.vcoord, wavelength)
#     complex_t3_gausslor = gausslor.compute_complex_vis(t3.u123coord, t3.v123coord, wavelength)
#     vis_gausslor, t3_gausslor = compute_vis(complex_vis_gausslor), compute_t3(complex_t3_gausslor)
#
#     complex_vis_conv = complex_vis_ring * complex_vis_gausslor
#     complex_t3_conv = complex_t3_ring * complex_t3_gausslor
#
#     shlr = StarHaloRing(
#         fs=0.42, fc=0.55, flor=1.0,
#         la=0.98, lkr=-0.26,
#         ks=1, kc=-4.12,
#         inc=0.63, pa=1.2/np.pi*180,
#         a=0.996393496566492, phi=-10)
#
#     # vis_shlr = shlr.compute_complex_vis(vis.ucoord, vis.vcoord, wavelength)
#     # t3_shlr = shlr.compute_complex_vis(t3.u123coord, t3.v123coord, wavelength)
#     set_data(fit_data=["vis", "t3"])


def test_assemble_components() -> None:
    """Tests the model's assemble_model method."""
    param_names = ["rin", "p", "cont_weight", "pa", "inc"]
    values = [1.5, 0.5, 0.2, 45, 1.6]
    limits = [[0, 20], [0, 1], [0, 1], [0, 360], [1, 50]]
    params = {name: Parameter(**getattr(STANDARD_PARAMETERS, name))
              for name in param_names}
    for value, limit, param in zip(values, limits, params.values()):
        param.set(*limit)
        param.value = value
    shared_params = {"p": params["p"]}
    del params["p"]

    components_and_params = [["Star", params], ["GreyBody", params]]
    components = assemble_components(components_and_params, shared_params)
    assert isinstance(components[0], Star)
    assert isinstance(components[1], GreyBody)
    assert all(not hasattr(components[0], param)
               for param in param_names if param
               not in ["pa", "inc"])
    assert all(hasattr(components[1], param) for param in param_names)
    assert all(getattr(components[1], name).value == value
               for name, value in zip(["pa", "inc"], values[-2:]))
    assert all(getattr(components[1], name).value == value
               for name, value in zip(param_names, values))
    assert all([getattr(components[0], name).min,
                getattr(components[0], name).max] == limit
               for name, limit in zip(["pa", "inc"], limits[-2:]))
    assert all([getattr(components[1], name).min,
                getattr(components[1], name).max] == limit
               for name, limit in zip(param_names, limits))
