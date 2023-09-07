import time
from pathlib import Path
from typing import Tuple, List

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ppdmod import fft
from ppdmod.options import OPTIONS
from ppdmod.utils import make_workbook, uniform_disk, uniform_disk_vis,\
    binary, binary_vis


RESOLUTION_FILE = Path("resolution.xlsx")
RESOLUTION_SHEET = "Binnings for 13 um"
WAVELENGTH_SHEET = "Resolutions for Wavelengths"
TIME_SHEET = "Execution Times of FFT"

make_workbook(
    RESOLUTION_FILE,
    {
        RESOLUTION_SHEET: ["Dimension [px]",
                           "Pixel Size [mas/px]",
                           "Frequency Spacing [m]",
                           "Max Frequency [m]",
                           "Binning Factor",
                           "Lambda/2B (B = 130 m) [mas]",
                           "Dimension (Pre-binning) [px]",
                           "Pixel Size (Pre-binning) [mas/px]"],
        WAVELENGTH_SHEET: ["Dimension [px]",
                           "Frequency Spacing [m]",
                           "Max Frequency [m]",
                           "Lambda/2B (B = 130 m) [mas]",
                           "Wavelength [um]",
                           "Pixel Size [mas/px]"
                           ],
        TIME_SHEET: ["Dimension [px]",
                     "Numpy FFT Time [s]",
                     "Numpy RFFT Time [s]",
                     "Scipy FFT Time [s]",
                     "Jax FFT Time [s]"]
    })


FFT_DIR = Path("fft")
if not FFT_DIR.exists():
    FFT_DIR.mkdir()


@pytest.fixture
def ucoord() -> u.m:
    """Sets the ucoord."""
    return np.linspace(100, 160, 60)*u.m


@pytest.fixture
def vcoord(ucoord: u.m) -> u.m:
    """Sets the vcoord."""
    return ucoord*0


@pytest.fixture
def u123coord() -> u.m:
    """Sets the ucoords for the closure phases."""
    coordinates = np.random.rand(3, 4)*100
    sign = np.random.randint(2, size=(3, 4))
    sign[sign == 0.] = -1
    return coordinates*sign*u.m


@pytest.fixture
def v123coord() -> u.m:
    """Sets the ucoords for the closure phases."""
    coordinates = np.random.rand(3, 4)*100
    sign = np.random.randint(2, size=(3, 4))
    sign[sign == 0.] = -1
    return coordinates*sign*u.m


@pytest.fixture
def pixel_size() -> u.mas:
    """Sets the pixel size."""
    return 0.1*u.mas


@pytest.fixture
def fluxes() -> Tuple[u.Quantity[u.Jy]]:
    """The fluxes of a binary."""
    return 5*u.Jy, 2*u.Jy


@pytest.fixture
def positions() -> Tuple[List[u.Quantity[u.mas]]]:
    """The positions of a binary."""
    return [5, 10]*u.mas, [-10, -10]*u.mas


@pytest.fixture
def wavelength() -> u.m:
    """A wavelenght grid."""
    return (13.000458e-6*u.m).to(u.um)

# TODO: Add tests for both the fft2 and rfft2 with the new modular system.


@pytest.mark.parametrize("backend", ["numpy", "scipy", "jax"])
def test_compute2Dfourier_transform(pixel_size: u.mas, backend: str,
                                    fluxes: Tuple[u.Quantity[u.Jy]]) -> None:
    """Tests the computation of the 2D fourier transform."""
    OPTIONS["fourier.backend"] = backend
    dim = 512
    ud = uniform_disk(pixel_size, dim, diameter=4*u.mas)
    rfft2_ud = fft.compute_real2Dfourier_transform(ud.value)
    
    assert rfft2_ud.shape == (ud.shape[0], ud.shape[1])

    _, (ax, bx, cx) = plt.subplots(1, 3)
    ax.imshow(ud.value)
    ax.set_title("Image space")
    ax.set_xlabel("dim [px]")
    bx.imshow(np.abs(rfft2_ud))
    bx.set_title("Magnitude")
    bx.set_xlabel("dim [px]")
    cx.imshow(np.angle(rfft2_ud))
    cx.set_title("Phase")
    cx.set_xlabel("dim [px]")
    plt.savefig(
        FFT_DIR / f"backend_{backend}_fourier_space_ud.pdf", format="pdf")
    plt.close()

    flux1, flux2 = fluxes
    position1, position2 = [0.5, 0.5]*u.mas, [-0.5, -0.5]*u.mas
    bin = binary(dim, pixel_size, flux1, flux2, position1, position2)
    rfft2_bin = fft.compute_real2Dfourier_transform(bin.value)
    assert rfft2_bin.shape == (bin.shape[0], bin.shape[1])

    _, (ax, bx, cx) = plt.subplots(1, 3)
    ax.imshow(bin.value)
    ax.set_title("Image space")
    ax.set_xlabel("dim [px]")
    bx.imshow(np.abs(rfft2_bin))
    bx.set_title("Magnitude")
    bx.set_xlabel("dim [px]")
    cx.imshow(np.angle(rfft2_bin))
    cx.set_title("Phase")
    cx.set_xlabel("dim [px]")
    plt.savefig(
        FFT_DIR / f"backend_{backend}_fourier_space_bin.pdf", format="pdf")
    plt.close()
    OPTIONS["fourier.backend"] = "numpy"


# TODO: Move these Todo's… Wrong file?
# TODO: Check that the Parameter and such actually gets the right values
# (even with the new scheme). Also the set data.
@pytest.mark.parametrize("dim, binning",
                         [(dim, binning) for binning in range(0, 11)
                          for dim in [2**power for power in range(7, 15)]])
def test_get_frequency_axis(dim: int, binning: int,
                            pixel_size: u.mas, wavelength: u.um) -> None:
    """Tests the frequency axis calculation and transformation."""
    OPTIONS["fourier.binning"] = binning
    frequency_axis = fft.get_frequency_axis(dim, pixel_size, wavelength)
    frequency_spacing = 1/(pixel_size.to(u.rad).value*dim)*wavelength.to(u.m)
    frequency_spacing /= 2**binning
    max_frequency = 0.5*dim*frequency_spacing
    lambda_over_2b = (wavelength.to(u.m).value/(2*130))*u.rad.to(u.mas)

    assert frequency_axis.unit == u.m
    assert frequency_axis.shape == (dim//2+1, )
    assert np.diff(frequency_axis)[0] == frequency_spacing
    assert frequency_axis.max() == max_frequency

    data = {"Dimension [px]": [dim],
            "Pixel Size [mas/px]": [pixel_size.value*2**binning],
            "Frequency Spacing [m]": np.around(frequency_spacing, 2),
            "Max Frequency [m]": np.around(max_frequency, 2),
            "Binning Factor": [binning],
            "Lambda/2B (B = 130 m) [mas]": [np.around(lambda_over_2b, 2)],
            "Dimension (Pre-binning) [px]": [dim*2**binning],
            "Pixel Size (Pre-binning) [mas/px]": [pixel_size.value]}

    if RESOLUTION_FILE.exists():
        df = pd.read_excel(RESOLUTION_FILE, sheet_name=RESOLUTION_SHEET)
        new_df = pd.DataFrame(data)
        df = pd.concat([df, new_df])
    else:
        df = pd.DataFrame(data)
    with pd.ExcelWriter(RESOLUTION_FILE, engine="openpyxl",
                        mode="a", if_sheet_exists="replace") as writer:
        df.to_excel(writer, sheet_name=RESOLUTION_SHEET, index=False)
    OPTIONS["fourier.binning"] = None


@pytest.mark.parametrize(
    "dim, wl",
    [(dim, wl) for wl in (range(5, 140, 5)*u.um)/10
     for dim in [2**power for power in range(7, 15)]])
def test_resolution_per_wavelength(dim: int,
                                   pixel_size: u.mas, wl: u.um) -> None:
    """Tests the frequency resolution per wavelength."""
    frequency_axis = fft.get_frequency_axis(dim, pixel_size, wl)
    frequency_spacing = 1/(pixel_size.to(u.rad).value*dim)*wl.to(u.m)
    max_frequency = 0.5*dim*frequency_spacing
    lambda_over_2b = (wl.to(u.m).value/(2*130))*u.rad.to(u.mas)
    assert frequency_axis.unit == u.m
    assert frequency_axis.shape == (dim//2+1, )
    assert np.diff(frequency_axis)[0] == frequency_spacing
    assert frequency_axis.max() == max_frequency

    data = {"Dimension [px]": [dim],
            "Frequency Spacing [m]": np.around(frequency_spacing, 2),
            "Max Frequency [m]": np.around(max_frequency, 2),
            "Lambda/2B (B = 130 m) [mas]": [np.around(lambda_over_2b, 2)],
            "Wavelength [um]": [wl.value],
            "Pixel Size [mas/px]": [pixel_size.value]
            }
    if RESOLUTION_FILE.exists():
        df = pd.read_excel(RESOLUTION_FILE, sheet_name=WAVELENGTH_SHEET)
        new_df = pd.DataFrame(data)
        df = pd.concat([df, new_df])
    else:
        df = pd.DataFrame(data)
    with pd.ExcelWriter(RESOLUTION_FILE, engine="openpyxl",
                        mode="a", if_sheet_exists="replace") as writer:
        df.to_excel(writer, sheet_name=WAVELENGTH_SHEET, index=False)


@pytest.mark.parametrize(
    "diameter, dim",
    [tuple([diameter, dim])
     for dim in [2**power for power in range(11, 13)]
     for diameter in [4, 10, 20]*u.mas])
def test_compile_full_fourier_from_real(
        diameter: u.mas, dim: int,
        pixel_size: u.mas,
        fluxes: Tuple[u.Quantity[u.Jy]],
        positions: Tuple[List[u.Quantity[u.mas]]]) -> None:
    """Tests the stiching together of the two parts of
    the real fourier transform into the full fourier transform."""
    flux1, flux2 = fluxes
    position1, position2 = positions

    ud = uniform_disk(pixel_size, dim, diameter=diameter).value
    bin = binary(dim, pixel_size, flux1, flux2, position1, position2).value

    OPTIONS["fourier.method"] = "real"
    fft_ud = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(ud)))
    fft_bin = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(bin)))
    rfft_ud = fft.compute_real2Dfourier_transform(ud)
    compiled_fft_ud = fft.compile_full_fourier_from_real(rfft_ud)
    rfft_bin = fft.compute_real2Dfourier_transform(bin)
    compiled_fft_bin = fft.compile_full_fourier_from_real(rfft_bin)

    assert np.allclose(fft_ud, compiled_fft_ud)
    assert np.allclose(fft_bin, compiled_fft_bin)
    OPTIONS["fourier.method"] = "complex"


def test_uv_coordinate_mirroring(pixel_size: u.mas) -> None:
    """Tests the uv coordinate mirroring."""
    dim = 512
    ud = uniform_disk(pixel_size, dim, 4*u.mas)
    frequency_axis_x = fft.get_frequency_axis(dim, pixel_size, 10*u.um, axis=1)
    frequency_axis_y = fft.get_frequency_axis(dim, pixel_size, 10*u.um, axis=0)
    rfft = fft.compute_real2Dfourier_transform(ud.value)

    _, (ax, bx) = plt.subplots(1, 2)
    ax.imshow(
        np.abs(rfft), extent=[0, frequency_axis_x.max().value,
                              frequency_axis_y.min().value,
                              frequency_axis_y.max().value])
    bx.imshow(
        np.abs(rfft), extent=[0, frequency_axis_x.max().value,
                              frequency_axis_y.min().value,
                              frequency_axis_y.max().value])

    array_configs = ["small", "medium", "large", "UTs"]
    colors = ["blue", "orange", "green", "red"]
    for config, color in zip(array_configs, colors):
        vis_ucoord, vis_vcoord =\
            np.load(f"data/uv_coords/{config}_uvcoords.npy")
        vis_index = np.where(vis_ucoord < 0)
        ax.scatter(vis_ucoord, vis_vcoord,
                   label=config, color=color, alpha=0.6)
        vis_ucoord_mirrored, vis_vcoord_mirrored, vis_conjugates =\
            fft.mirror_uv_coords(vis_ucoord, vis_vcoord)
        ax.scatter(vis_ucoord_mirrored,
                   vis_vcoord_mirrored, marker="x", color=color)

        assert all(vis_index[0] == np.where(vis_conjugates)[0])
        assert all(vis_ucoord[vis_index] == -vis_ucoord_mirrored[vis_index])
        assert all(vis_vcoord[vis_index] == -vis_vcoord_mirrored[vis_index])
        assert vis_ucoord.size == vis_ucoord_mirrored.size
        assert vis_vcoord.size == vis_vcoord_mirrored.size

        cphases_ucoord, cphases_vcoord = np.load(
            f"data/uv_coords/{config}_uv123coords.npy")
        bx.scatter(cphases_ucoord.flatten(), cphases_vcoord.flatten(),
                   label=config, color=color, alpha=0.6)
        cphases_ucoord_mirrored, cphases_vcoord_mirrored, cphase_conjugates =\
            fft.mirror_uv_coords(cphases_ucoord, cphases_vcoord)
        bx.scatter(cphases_ucoord_mirrored.flatten(),
                   cphases_vcoord_mirrored.flatten(),
                   color=color, marker="x")

        cphase_index = np.where(cphases_ucoord < 0)
        assert all(cphase_index[0] == np.where(cphase_conjugates)[0])
        assert all(cphases_ucoord[cphase_index] ==
                   -cphases_ucoord_mirrored[cphase_index])
        assert all(cphases_vcoord[cphase_index] ==
                   -cphases_vcoord_mirrored[cphase_index])
        assert cphases_ucoord.size == cphases_ucoord_mirrored.size
        assert cphases_vcoord.size == cphases_vcoord_mirrored.size

    ax.set_xlim([-200, 200])
    ax.set_ylim([-200, 200])
    ax.set_title("UV Coordinates Mirrored")
    ax.set_xlabel("u (m)")
    ax.set_ylabel("v (m)")
    ax.legend(loc="upper left", fontsize="small")

    bx.set_title("UV Triangles Mirrored")
    bx.set_xlabel("u (m)")
    bx.set_ylabel("v (m)")
    bx.set_xlim([-200, 200])
    bx.set_ylim([-200, 200])
    bx.legend(loc="upper left", fontsize="small")
    plt.savefig(FFT_DIR / "uv_coords_mirrored.pdf", format="pdf")
    plt.close()


# FIXME: Angle of the interpolated values is the negative of the
# calculated value?
@pytest.mark.parametrize(
    "diameter, dim",
    [tuple([diameter, dim])
     for dim in [2**power for power in range(11, 13)]
     for diameter in [4, 10, 20]*u.mas])
def test_vis_interpolation(diameter: u.mas, dim: float,
                           ucoord: u.m, vcoord: u.m,
                           pixel_size: u.mas, wavelength: u.um,
                           fluxes: Tuple[u.Quantity[u.Jy]],
                           positions: Tuple[List[u.Quantity[u.mas]]]
                           ) -> None:
    """This tests the interpolation of the Fourier transform,
    but more importantly, implicitly the unit conversion of the
    frequency axis for the visibilitites/correlated fluxes."""
    flux1, flux2 = fluxes
    position1, position2 = positions
    vis_ud = uniform_disk_vis(diameter, ucoord, vcoord, wavelength)
    vis_bin = binary_vis(
        flux1, flux2, ucoord, vcoord, position1, position2, wavelength)

    rfft_ud = fft.compute_real2Dfourier_transform(
        uniform_disk(pixel_size, dim, diameter=diameter))
    rfft_bin = fft.compute_real2Dfourier_transform(
        binary(dim, pixel_size, flux1, flux2, position1, position2))

    interpolated_ud = fft.interpolate_coordinates(
        rfft_ud, dim, pixel_size, ucoord, vcoord, wavelength)
    interpolated_bin = fft.interpolate_coordinates(
        rfft_bin, dim, pixel_size, ucoord, vcoord, wavelength)

    interpolated_ud /= rfft_ud.max()
    interpolated_ud = np.real(interpolated_ud)
    assert np.allclose(vis_ud, interpolated_ud, atol=1e-2)
    assert np.allclose(np.abs(vis_bin),
                       np.abs(interpolated_bin), atol=1e0)
    assert np.allclose(np.abs(np.angle(vis_bin)),
                       np.abs(np.angle(interpolated_bin)), atol=1e-2)

    for config in ["small", "medium", "large", "UTs"]:
        ucoord, vcoord =\
            np.load(f"data/uv_coords/{config}_uvcoords.npy")
        ucoord, vcoord = ucoord*u.m, vcoord*u.m
        vis_ud = uniform_disk_vis(diameter, ucoord, vcoord, wavelength)
        vis_bin = binary_vis(
            flux1, flux2, ucoord, vcoord, position1, position2, wavelength)
        interpolated_ud = fft.interpolate_coordinates(
            rfft_ud, dim, pixel_size, ucoord, vcoord, wavelength)
        interpolated_bin = fft.interpolate_coordinates(
            rfft_bin, dim, pixel_size, ucoord, vcoord, wavelength)
        interpolated_ud /= rfft_ud.max()
        interpolated_ud = np.real(interpolated_ud)
        assert np.allclose(vis_ud, interpolated_ud, atol=1e-2)
        assert np.allclose(np.abs(vis_bin),
                           np.abs(interpolated_bin), atol=1e0)
        assert np.allclose(np.abs(np.angle(vis_bin)),
                           np.abs(np.angle(interpolated_bin)), atol=1e-2)


@pytest.mark.parametrize(
    "diameter, dim",
    [tuple([diameter, dim])
     for dim in [1024, 2048, 4096]
     for diameter in [4, 10, 20]*u.mas])
def test_cphases_interpolation(diameter: u.mas, dim: float,
                               u123coord: u.m, v123coord: u.m,
                               pixel_size: u.mas, wavelength: u.um,
                               fluxes: Tuple[u.Quantity[u.Jy]],
                               positions: Tuple[List[u.Quantity[u.mas]]]
                               ) -> None:
    """Tests the interpolation of the closure phases."""
    flux1, flux2 = fluxes
    position1, position2 = positions
    rfft_ud = fft.compute_real2Dfourier_transform(
        uniform_disk(pixel_size, dim, diameter=diameter))
    rfft_bin = fft.compute_real2Dfourier_transform(
        binary(dim, pixel_size, flux1, flux2, position1, position2))

    interpolated_ud = fft.interpolate_coordinates(
        rfft_ud, dim, pixel_size, u123coord, v123coord, wavelength)
    interpolated_ud =\
        np.real(np.product(interpolated_ud/rfft_ud.max(), axis=1))

    interpolated_bin = fft.interpolate_coordinates(
        rfft_bin, dim, pixel_size, u123coord, v123coord, wavelength)
    interpolated_bin = np.angle(np.product(interpolated_bin, axis=1))

    cphase_ud, cphase_bin = [], []
    for ucoord, vcoord in zip(u123coord, v123coord):
        tmp_cphase_bin = binary_vis(flux1, flux2,
                                    ucoord, vcoord,
                                    position1, position2,
                                    wavelength)
        tmp_cphase_ud = uniform_disk_vis(diameter, ucoord, vcoord, wavelength)
        cphase_ud.append(tmp_cphase_ud)
        cphase_bin.append(tmp_cphase_bin)

    cphase_ud = np.product(cphase_ud, axis=0)
    cphase_bin = np.angle(np.product(cphase_bin, axis=0))

    # NOTE: Resolution is too low at 1024 to successfully fit a binary...
    if dim == 1024:
        assert np.allclose(cphase_ud, interpolated_ud, atol=1e-1)
        assert np.allclose(np.abs(cphase_bin),
                           np.abs(np.angle(interpolated_bin)), atol=1e1)
    else:
        assert np.allclose(cphase_ud, interpolated_ud, atol=1e-2)
        assert np.allclose(np.abs(cphase_bin), np.abs(interpolated_bin),
                           atol=1e-2)

    for config in ["small", "medium", "large", "UTs"]:
        cphases_ucoord, cphases_vcoord = np.load(
            f"data/uv_coords/{config}_uv123coords.npy")
        interpolated_bin = fft.interpolate_coordinates(
            rfft_bin, dim, pixel_size, u123coord, v123coord, wavelength)
        interpolated_bin = np.angle(np.product(interpolated_bin, axis=1))

        cphase_ud, cphase_bin = [], []
        for ucoord, vcoord in zip(u123coord, v123coord):
            tmp_cphase_bin = binary_vis(flux1, flux2,
                                        ucoord, vcoord,
                                        position1, position2,
                                        wavelength)
            tmp_cphase_ud = uniform_disk_vis(diameter, ucoord,
                                             vcoord, wavelength)
            cphase_ud.append(tmp_cphase_ud)
            cphase_bin.append(tmp_cphase_bin)

        cphase_ud = np.product(cphase_ud, axis=0)
        cphase_bin = np.angle(np.product(cphase_bin, axis=0))

        # NOTE: Resolution is too low at 1024 to successfully fit a binary...
        if dim == 1024:
            assert np.allclose(cphase_ud, interpolated_ud, atol=1e-1)
            assert np.allclose(np.abs(cphase_bin),
                               np.abs(np.angle(interpolated_bin)), atol=1e1)
        else:
            assert np.allclose(cphase_ud, interpolated_ud, atol=1e-2)
            assert np.allclose(np.abs(cphase_bin), np.abs(interpolated_bin),
                               atol=1e-2)


@pytest.mark.parametrize(
    "diameter, dim",
    [tuple([diameter, dim])
     for dim in [1024, 2048, 4096]
     for diameter in [1, 2, 3, 4, 5, 6, 10, 20]*u.mas])
def test_numpy_vs_scipy_vs_fftw(diameter: u.mas, dim: float,
                                pixel_size: u.mas,
                                fluxes: Tuple[u.Quantity[u.Jy]],
                                positions: Tuple[List[u.Quantity[u.mas]]]
                                ) -> None:
    """Compares the numpy vs the pyfftw implementation of the
    Fourier transform."""
    flux1, flux2 = fluxes
    position1, position2 = positions
    image_ud = uniform_disk(pixel_size, dim, diameter=diameter)
    image_bin = binary(dim, pixel_size, flux1, flux2, position1, position2)

    st_numpy_fft = time.time()
    _ = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image_ud)))
    et_numpy_fft = time.time()-st_numpy_fft

    OPTIONS["fourier.backend"] = "numpy"
    OPTIONS["fourier.method"] = "real"
    st_numpy_fft = time.time()
    numpy_fft_ud = fft.compute_real2Dfourier_transform(image_ud)
    et_numpy_fft = time.time()-st_numpy_fft
    numpy_fft_bin = fft.compute_real2Dfourier_transform(image_bin)
    OPTIONS["fourier.method"] = "complex"

    OPTIONS["fourier.backend"] = "numpy"
    st_scipy_fft = time.time()
    scipy_fft_ud = fft.compute_real2Dfourier_transform(image_ud)
    et_scipy_fft = time.time()-st_scipy_fft
    scipy_fft_bin = fft.compute_real2Dfourier_transform(image_bin)

    OPTIONS["fourier.backend"] = "jax"
    st_jax_fft = time.time()
    jax_fft_ud = fft.compute_real2Dfourier_transform(image_ud)
    et_jax_fft = time.time()-st_jax_fft
    jax_fft_bin = fft.compute_real2Dfourier_transform(image_bin)

    _, axarr = plt.subplots(4, 3)
    axarr = axarr.flatten()
    for name, rfft in [("Numpy UD", numpy_fft_ud),
                       ("Numpy Bin", numpy_fft_bin),
                       ("Scipy UD", scipy_fft_ud),
                       ("Scipy Bin", scipy_fft_bin),
                       ("Jax UD", jax_fft_ud),
                       ("Jax Bin", jax_fft_bin)]:
        axarr[0].imshow(np.abs(rfft))
        axarr[0].set_title(f"Magnitude {name}")
        axarr[0].set_xlabel("dim [px]")
        axarr[1].imshow(np.angle(rfft))
        axarr[1].set_title(f"Phase {name}")
        axarr[1].set_xlabel("dim [px]")

    plt.savefig(
        FFT_DIR / f"numpy_vs_pyfftw_dim{dim}_dia{diameter.value}.pdf",
        format="pdf")
    plt.close()

    data = {"Dimension [px]": [dim],
            "Numpy FFT Time [s]": [et_numpy_fft],
            "Numpy RFFT Time [s]": [et_numpy_fft],
            "Scipy RFFT Time [s]": [et_scipy_fft],
            "Jax RFFT Time [s]": [et_jax_fft]}

    if RESOLUTION_FILE.exists():
        df = pd.read_excel(RESOLUTION_FILE, sheet_name=TIME_SHEET)
        new_df = pd.DataFrame(data)
        df = pd.concat([df, new_df])
    else:
        df = pd.DataFrame(data)
    with pd.ExcelWriter(RESOLUTION_FILE, engine="openpyxl",
                        mode="a", if_sheet_exists="replace") as writer:
        df.to_excel(writer, sheet_name=TIME_SHEET, index=False)

    assert np.allclose(numpy_fft_ud, scipy_fft_ud, atol=1e-2)
    assert np.allclose(numpy_fft_ud, jax_fft_ud, atol=1e-2)
    assert np.allclose(numpy_fft_bin, scipy_fft_bin)
    assert np.allclose(numpy_fft_bin, jax_fft_bin)
    OPTIONS["fourier.backend"] = "numpy"
