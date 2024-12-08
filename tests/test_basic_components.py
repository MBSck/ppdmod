import time
from pathlib import Path
from typing import List

import astropy.units as u
import numpy as np
import pandas as pd
import pytest

from ppdmod import utils
from ppdmod.basic_components import (
    AsymGreyBody,
    PointSource,
    Ring,
    Star,
    TempGradient,
)
from ppdmod.data import ReadoutFits, set_data
from ppdmod.options import OPTIONS
from ppdmod.parameter import Parameter
from ppdmod.utils import (
    compute_t3,
    compute_vis,
    get_opacity,
    get_t3_from_vis,
    load_data,
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
    return Star(dim=512, dist=145, eff_temp=7800, eff_radius=1.8)


@pytest.fixture
def ring() -> Ring:
    """Initializes a ring component."""
    return Ring(dim=512, diam=5, width=1)


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
        rin=0.5,
        q=0.5,
        temp0=1500,
        dist=148.3,
        sigma0=2000,
        pixel_size=0.1,
        p=0.5,
    )
    temp_grad.optically_thick = True
    temp_grad.asymmetric = True
    return temp_grad


@pytest.fixture
def kappa_abs() -> Parameter:
    weights = np.array([0.0, 73.2, 8.6, 0.0, 0.6, 14.2, 2.4, 1.0]) / 100
    names = ["pyroxene", "forsterite", "enstatite", "silica"]
    grid, value = get_opacity(DATA_DIR / "opacities", weights, names, "boekel")
    return Parameter(grid=grid, value=value, base="kappa_abs")


@pytest.fixture
def kappa_cont() -> Parameter:
    cont_opacity_file = (
        DATA_DIR / "opacities" / "optool" / "preibisch_amorph_c_rv0.1.npy"
    )
    grid, value = np.load(cont_opacity_file)
    return Parameter(grid=grid, value=value, base="kappa_cont")


@pytest.fixture
def star_flux() -> Parameter:
    grid, value = load_data(
        DATA_DIR / "flux" / "hd142527" / "HD142527_stellar_model.txt"
    )
    return Parameter(grid=grid, value=value, base="f")


@pytest.fixture
def luminosity() -> float:
    return 1.35


@pytest.fixture
def eff_radius(luminosity: float, eff_temp: int) -> float:
    return utils.compute_stellar_radius(10**luminosity, eff_temp).value


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
#     vis = point_source.compute_complex_vis(
#         READOUT.vis.ucoord, READOUT.vis.vcoord, wavelength
#     )
#     assert vis.shape == (wavelength.size, READOUT.vis.ucoord.size)
#
#
# @pytest.mark.parametrize(
#     "wl, dim",
#     [
#         (u.Quantity([wl], unit=u.um), dim)
#         for dim in DIMENSION
#         for wl in [8, 9, 10, 11] * u.um
#     ],
# )
# def test_point_source_image(point_source: Star, dim: int, wl: u.um) -> None:
#     """Tests the point source's image calculation."""
#     image = point_source.compute_image(dim, 0.1 * u.mas, wl)
#     point_source_dir = Path("images/point_source")
#     point_source_dir.mkdir(exist_ok=True, parents=True)
#     centre = dim // 2
#     plt.imshow(image[0])
#     plt.xlim(centre - 20, centre + 20)
#     plt.ylim(centre - 20, centre + 20)
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
# # TODO: Fix this? What is the problem
# # TODO: Make this for multiple wavelengths at the same time
# @pytest.mark.parametrize(
#     "wl, dim",
#     [
#         (u.Quantity([wl], unit=u.um), dim)
#         for dim in DIMENSION
#         for wl in [8, 9, 10, 11] * u.um
#     ],
# )
# def test_star_image(star: Star, dim: int, wl: u.um) -> None:
#     """Tests the star's image calculation."""
#     image = star.compute_image(dim, 0.1 * u.mas, wl)
#
#     star_dir = Path("images/star")
#     star_dir.mkdir(exist_ok=True, parents=True)
#
#     centre = dim // 2
#
#     plt.imshow(image[0])
#     plt.xlim(centre - 20, centre + 20)
#     plt.ylim(centre - 20, centre + 20)
#     plt.savefig(star_dir / f"dim{dim}_wl{wl.value}_star_image.pdf")
#     plt.close()
#
#     assert len(image[image != 0]) == 4
#     assert image.shape == (1, dim, dim)
#     assert np.max(image) < 0.1


# def test_uniform_ring_init(ring: Ring) -> None:
#     """Tests the ring's initialization."""
#     assert "rin" in vars(ring).keys()
#     assert "rout" in vars(ring).keys()
#     assert "width" in vars(ring).keys()


# @pytest.mark.parametrize(
#     "globs",
#     [
#         ["PION*"],
#         ["GRAV*"],
#         ["*_L_*"],
#         ["*_N_*"],
#         ["PION*", "GRAV*"],
#         ["PION*", "GRAV*", "_L_"],
#         ["PION*", "GRAV*", "_L_", "_N_"],
#     ],
# )
# def test_uv_coord_assignment(globs: List[str]) -> None:
#     fits_files = []
#     for glob in globs:
#         fits_files.extend(list((DATA_DIR / "fits" / "hd142527").glob(f"{glob}.fits")))
#
#     binning = OPTIONS.data.binning
#     wavelengths = {
#         "hband": [1.7] * u.um,
#         "kband": [2.15] * u.um,
#         "lband": utils.windowed_linspace(3.1, 3.4, binning.lband.value) * u.um,
#         "mband": utils.windowed_linspace(4.7, 4.9, binning.mband.value) * u.um,
#         "nband": utils.windowed_linspace(8, 13, binning.nband.value) * u.um,
#     }
#     wavelengths = np.concatenate(
#         (
#             wavelengths["hband"],
#             wavelengths["kband"],
#             wavelengths["lband"],
#             wavelengths["mband"],
#             wavelengths["nband"],
#         )
#     )
#     data = set_data(fits_files, wavelengths=wavelengths)
#     t3_u123coord = np.array(
#         [data.vis.ucoord[:, ind] for ind in OPTIONS.data.t3.sta_vis_index.T]
#     ).reshape(3, -1)
#     t3_v123coord = np.array(
#         [data.vis.vcoord[:, ind] for ind in OPTIONS.data.t3.sta_vis_index.T]
#     ).reshape(3, -1)
#
#     mask = data.t3.sta_conj_flag.T
#     t3_u123coord[mask] = -t3_u123coord[mask]
#     t3_v123coord[mask] = -t3_v123coord[mask]
#
#     try:
#         assert np.allclose(t3_u123coord, data.t3.u123coord[:, 1:], atol=1e-2)
#     except AssertionError:
#         assert np.all(np.abs(t3_u123coord - data.t3.u123coord[:, 1:]) < 2)
#
#     try:
#         assert np.allclose(t3_v123coord, data.t3.v123coord[:, 1:], atol=1e-2)
#     except AssertionError:
#         assert np.all(np.abs(t3_v123coord - data.t3.v123coord[:, 1:]) < 2)
#
#
# @pytest.mark.parametrize(
#     "array, wl, rin, width, cinc, pa, mod_amps",
#     [
#         ("uts", 3.5, 1.5, 0.25, 1, 0, None),
#         ("uts", 3.5, 1.5, 0.25, 0.5, 285, None),
#         ("uts", 3.5, 1.5, 0.25, 0.63, 68, [(-0.18, 0.98)]),
#     ],
# )
# def test_ring_compute_vis(
#     array: str,
#     wl: u.um,
#     rin: u.mas,
#     width: u.mas,
#     cinc: float,
#     pa: u.deg,
#     mod_amps: List[float],
# ) -> None:
#     """Tests the calculation of the ring's visibilities."""
#     if isinstance(wl, list):
#         wavelength = wl * u.um
#     else:
#         wavelength = [wl] * u.um
#
#     asymmetric, mod_dict = False, {}
#     if mod_amps is not None:
#         asymmetric = True
#         OPTIONS.model.modulation = len(mod_amps)
#         mod_amps = mod_amps[: OPTIONS.model.modulation]
#         mod_dict = {f"c{i+1}": amp[0] for i, amp in enumerate(mod_amps)}
#         mod_dict.update({f"s{i+1}": amp[1] for i, amp in enumerate(mod_amps)})
#
#     params_dict = {"rin": rin, "width": width, "cinc": cinc, "pa": pa}
#     params_dict.update(mod_dict)
#     param_labels = [
#         f"{key}{str(value).replace('.', '')}" for key, value in params_dict.items()
#     ]
#     fits_file = DATA_DIR / "aspro" / f"{'_'.join(['Ring', *param_labels, array])}.fits"
#     data = set_data([fits_file], wavelengths=wavelength, fit_data=["vis", "t3"])
#
#     ring = Ring(
#         rin=rin,
#         width=width,
#         cinc=cinc,
#         pa=pa,
#         has_outer_radius=False,
#         thin=False,
#         asymmetric=asymmetric,
#         **mod_dict,
#     )
#
#     vis, t3 = data.vis2 if "vis2" in OPTIONS.fit.data else data.vis, data.t3
#     complex_vis = ring.compute_complex_vis(vis.ucoord, vis.vcoord, wavelength)
#     vis_ring, t3_ring = compute_vis(complex_vis), get_t3_from_vis(complex_vis)
#     if "vis2" in OPTIONS.fit.data:
#         vis_ring *= vis_ring
#
#     vis_ring, t3_ring = vis_ring[:, 1:7], t3_ring[:, :4]
#
#     assert vis_ring.shape == (wavelength.size, 6)
#     assert np.allclose(vis.value[:, :6], vis_ring, atol=1e-2)
#
#     assert t3_ring.shape == (wavelength.size, 4)
#     assert np.allclose(t3.value[:, :4], t3_ring, atol=1e0)
#
#     set_data()
#     OPTIONS.model.modulation = 1
#
#
# @pytest.mark.parametrize("grid_type", ["linear", "logarithmic"])
# def test_temp_gradient_compute_grid(
#     temp_gradient: TempGradient, grid_type: str
# ) -> None:
#     """Tests the hankel component's grid calculation."""
#     OPTIONS.model.gridtype = grid_type
#     radius = temp_gradient.compute_internal_grid()
#     assert radius.unit == u.au
#     assert radius.shape == (temp_gradient.dim(),)
#     assert (
#         radius[0].value == temp_gradient.rin.value
#         and radius[-1].value == temp_gradient.rout.value
#     )
#
#     OPTIONS.model.gridtype = "logarithmic"
#
#
# def test_temp_gradient_flux(temp_gradient: TempGradient, wavelength: u.um) -> None:
#     """Tests the calculation of the total flux."""
#     flux = temp_gradient.compute_flux(wavelength)
#     assert flux.shape == (wavelength.size, 1)
#
#
# @pytest.mark.parametrize("dim", [4096, 2096, 1024, 512, 256, 128, 64, 32])
# def test_temp_gradient_resolution(
#     temp_gradient: TempGradient, dim: int, wavelength: u.um
# ) -> None:
#     """Tests the hankel component's resolution."""
#     temp_gradient.dim.value = dim
#     temp_gradient.optically_thick = True
#     temp_gradient.asymmetric = False
#
#     OPTIONS.model.modulation = 1
#     start_time_vis = time.perf_counter()
#     _ = temp_gradient.compute_complex_vis(
#         READOUT.vis2.ucoord, READOUT.vis2.vcoord, wavelength
#     )
#     end_time_vis = time.perf_counter() - start_time_vis
#
#     start_time_cphase = time.perf_counter()
#     _ = temp_gradient.compute_complex_vis(
#         READOUT.t3.u123coord, READOUT.t3.v123coord, wavelength
#     )
#     end_time_cphase = time.perf_counter() - start_time_cphase
#
#     vis_data = {"Dimension (px)": [dim], "Computation Time (s)": [end_time_vis]}
#
#     t3_data = {"Dimension (px)": [dim], "Computation Time (s)": [end_time_cphase]}
#
#     if CALCULATION_FILE.exists():
#         df = pd.read_excel(CALCULATION_FILE, sheet_name="Vis")
#         new_df = pd.DataFrame(vis_data)
#         df = pd.concat([df, new_df])
#     else:
#         df = pd.DataFrame(vis_data)
#
#     with pd.ExcelWriter(
#         CALCULATION_FILE, engine="openpyxl", mode="a", if_sheet_exists="replace"
#     ) as writer:
#         df.to_excel(writer, sheet_name="Vis", index=False)
#
#     if CALCULATION_FILE.exists():
#         df = pd.read_excel(CALCULATION_FILE, sheet_name="T3")
#         new_df = pd.DataFrame(t3_data)
#         df = pd.concat([df, new_df])
#     else:
#         df = pd.DataFrame(t3_data)
#
#     with pd.ExcelWriter(
#         CALCULATION_FILE, engine="openpyxl", mode="a", if_sheet_exists="replace"
#     ) as writer:
#         df.to_excel(writer, sheet_name="T3", index=False)
#
#     OPTIONS.model.modulation = 0
#     temp_gradient.optically_thick = False
#     temp_gradient.asymmetric = False


def test_t3_calculation(kappa_abs: Parameter, kappa_cont: Parameter) -> None:
    """Tests the calculation of the ring's visibilities."""
    binning = OPTIONS.data.binning
    wavelengths = {
        "hband": [1.7] * u.um,
        "kband": [2.15] * u.um,
        "lband": utils.windowed_linspace(3.1, 3.4, binning.lband.value) * u.um,
        "mband": utils.windowed_linspace(4.7, 4.9, binning.mband.value) * u.um,
        "nband": utils.windowed_linspace(8, 13, binning.nband.value) * u.um,
    }
    wavelengths = np.concatenate(
        (
            wavelengths["hband"],
            wavelengths["kband"],
            wavelengths["lband"],
            wavelengths["mband"],
            wavelengths["nband"],
        )
    )

    data = set_data(
        list((DATA_DIR / "fits" / "hd142527").glob("*.fits")),
        wavelengths=wavelengths,
    )

    disc = AsymGreyBody(
        dim=32,
        dist=158.51,
        eff_temp=6750,
        eff_radius=3.46,
        kappa_abs=kappa_abs,
        kappa_cont=kappa_cont,
        pa=33,
        cinc=0.8,
        rin=0.1,
        rout=1.5,
        p=0.6,
        sigma0=1e-3,
        c1=0.5,
        s1=-1,
    )

    vis, t3 = data.vis2 if "vis2" in OPTIONS.fit.data else data.vis, data.t3
    complex_vis = disc.compute_complex_vis(vis.ucoord, vis.vcoord, wavelengths)
    complex_vis_t3 = disc.compute_complex_vis(t3.u123coord, t3.v123coord, wavelengths)
    t3_ring_calc = compute_t3(complex_vis_t3)
    t3_ring_get = get_t3_from_vis(complex_vis)

    assert np.allclose(t3_ring_calc[:, 1:], t3_ring_get, atol=1e0)

    set_data()
    OPTIONS.model.modulation = 1
