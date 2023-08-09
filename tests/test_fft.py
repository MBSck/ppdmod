from pathlib import Path
from typing import Tuple, List

import astropy.units as u
import numpy as np
import pandas as pd
import pytest

from ppdmod.fft import compute_2Dfourier_transform, get_frequency_axis,\
    interpolate_coordinates
from ppdmod.options import OPTIONS
from ppdmod.utils import uniform_disk, uniform_disk_vis,\
    binary, binary_vis


RESOLUTION_AND_FLUX_DIR = Path("resolution_and_flux")
if not RESOLUTION_AND_FLUX_DIR.exists():
    RESOLUTION_AND_FLUX_DIR.mkdir()

RESOLUTION_WAVELENGTH_FILE = RESOLUTION_AND_FLUX_DIR / "frequency_resolution_wavelength.csv"
if RESOLUTION_WAVELENGTH_FILE.exists():
    RESOLUTION_WAVELENGTH_FILE.unlink()

RESOLUTION_FILE = RESOLUTION_AND_FLUX_DIR / "frequency_resolution.csv"
if RESOLUTION_FILE.exists():
    RESOLUTION_FILE.unlink()


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
def fluxes() -> Tuple[u.Jy]:
    """The fluxes of a binary."""
    return 5*u.Jy, 2*u.Jy


@pytest.fixture
def positions() -> Tuple[List[u.mas]]:
    """The positions of a binary."""
    return [5, 10]*u.mas, [-10, -10]*u.mas


@pytest.fixture
def wavelength() -> u.um:
    """Sets the wavelength."""
    return 12*u.um


def test_compute2Dfourier_transform(pixel_size: u.mas) -> None:
    """Tests the computation of the 2D fourier transform."""
    ud = uniform_disk(pixel_size, 512, diameter=4*u.mas)
    ft = compute_2Dfourier_transform(ud)
    assert ft.shape == ud.shape
    assert ft.dtype == np.complex128


# TODO: Add calculation also for different wavelengths, different tables or so?
# TODO: Test input for both wavelength in meter and um?
# TODO: Check that the Parameter and such actually gets the right values (even with
# the new scheme). Also the set data.
@pytest.mark.parametrize("dim, binning",
                         [(dim, binning) for binning in range(0, 11)
                          for dim in [2**power for power in range(7, 15)]])
def test_get_frequency_axis(dim: int, binning: int,
                            pixel_size: u.mas, wavelength: u.um) -> None:
    """Tests the frequency axis calculation and transformation."""
    OPTIONS["fourier.binning"] = binning
    frequency_axis = get_frequency_axis(dim, pixel_size, wavelength)
    frequency_spacing = 1/(pixel_size.to(u.rad).value*dim)*wavelength.to(u.m)
    frequency_spacing /= 2**binning
    assert frequency_axis.unit == u.m
    assert frequency_axis.shape == (dim, )
    assert np.isclose(np.diff(frequency_axis)[0], frequency_spacing)
    assert -frequency_axis.min() == 0.5*dim*frequency_spacing

    binning_str = 0 if binning is None else binning
    data = {"Dimension [px]": [dim], "Binning Factor": [binning_str],
            "Frequency Spacing [m]": np.around(frequency_spacing, 2),
            "Pre-binning [px]": [dim*2**binning_str]}
    try:
        df = pd.read_csv(RESOLUTION_FILE)
        new_df = pd.DataFrame(data)
        df = pd.concat([df, new_df])
    except FileNotFoundError:
        df = pd.DataFrame(data)
    df.to_csv(RESOLUTION_FILE, index=False)


@pytest.mark.parametrize("dim, wl",
                         [(dim, wl) for dim in [2** power for power in range(7, 15)]
                          for wl in range(2, 14, 2)*u.um])
def test_resolution_per_wavelength(dim: int, pixel_size: u.mas, wl: u.um) -> None:
    """Tests the frequency axis calculation and transformation."""
    OPTIONS["fourier.binning"] = None
    frequency_axis = get_frequency_axis(dim, pixel_size, wl)
    frequency_spacing = 1/(pixel_size.to(u.rad).value*dim)*wl.to(u.m)
    assert frequency_axis.unit == u.m
    assert frequency_axis.shape == (dim, )
    assert np.isclose(np.diff(frequency_axis)[0], frequency_spacing)
    assert -frequency_axis.min() == 0.5*dim*frequency_spacing

    data = {"Dimension [px]": [dim],
            "Frequency Spacing [m]": np.around(frequency_spacing, 2),
            "Wavelength [um]": [wl]}
    try:
        df = pd.read_csv(RESOLUTION_WAVELENGTH_FILE)
        new_df = pd.DataFrame(data)
        df = pd.concat([df, new_df])
    except FileNotFoundError:
        df = pd.DataFrame(data)
    df.to_csv(RESOLUTION_WAVELENGTH_FILE, index=False)


@pytest.mark.parametrize(
    "diameter, dim",
    [tuple([diameter, dim])
     for dim in [1024, 2048, 4096]
     for diameter in [4, 10, 20]*u.mas])
def test_cphases_interpolation(diameter: u.mas, dim: float,
                               u123coord: u.m, v123coord: u.m,
                               pixel_size: u.mas, wavelength: u.um,
                               fluxes: Tuple[u.Jy], positions: Tuple[List[u.mas]]
                               ) -> None:
    """Tests the interpolation of the closure phases."""
    OPTIONS["fourier.binning"] = None
    flux1, flux2 = fluxes
    position1, position2 = positions

    ft_ud = compute_2Dfourier_transform(uniform_disk(pixel_size, dim,
                                                  diameter=diameter))
    interpolated_ud = interpolate_coordinates(
        ft_ud, dim, pixel_size, u123coord, v123coord, wavelength)
    interpolated_ud = np.product(
        interpolated_ud/ft_ud[dim//2, dim//2], axis=1)
    interpolated_ud = np.real(interpolated_ud)

    ft_bin = compute_2Dfourier_transform(
        binary(dim, pixel_size, flux1, flux2, position1, position2))
    interpolated_bin = interpolate_coordinates(ft_bin, dim,
                                               pixel_size,
                                               u123coord, v123coord,
                                               wavelength)
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
        assert np.allclose(np.abs(cphase_bin), np.abs(interpolated_bin), atol=1e-2)


# TODO: Angle of the interpolated values is the negative of the calculated value?
@pytest.mark.parametrize(
    "diameter, dim",
    [tuple([diameter, dim])
     for dim in [2**power for power in range(11, 13)]
     for diameter in [4, 10, 20]*u.mas])
def test_vis_interpolation(diameter: u.mas, dim: float,
                           ucoord: u.m, vcoord: u.m,
                           pixel_size: u.mas, wavelength: u.um,
                           fluxes: Tuple[u.Jy], positions: Tuple[List[u.mas]]
                           ) -> None:
    """This tests the interpolation of the Fourier transform,
    but more importantly, implicitly the unit conversion of the
    frequency axis for the visibilitites/correlated fluxes."""
    OPTIONS["fourier.binning"] = None
    flux1, flux2 = fluxes
    position1, position2 = positions
    ft_ud = compute_2Dfourier_transform(uniform_disk(pixel_size, dim,
                                                     diameter=diameter))
    vis_ud = uniform_disk_vis(diameter, ucoord, vcoord, wavelength)
    ft_bin = compute_2Dfourier_transform(
        binary(dim, pixel_size, flux1, flux2, position1, position2))
    vis_bin = binary_vis(flux1, flux2,
                         ucoord, vcoord,
                         position1, position2,
                         wavelength)
    interpolated_ud = interpolate_coordinates(ft_ud, dim,
                                              pixel_size,
                                              ucoord, vcoord,
                                              wavelength)
    interpolated_bin = interpolate_coordinates(ft_bin, dim,
                                               pixel_size,
                                               ucoord, vcoord,
                                               wavelength)
    interpolated_ud /= ft_ud[dim//2, dim//2]
    interpolated_ud = np.real(interpolated_ud)
    assert np.allclose(vis_ud, interpolated_ud, atol=1e-2)
    assert np.allclose(np.abs(vis_bin),
                       np.abs(interpolated_bin), atol=1e0)
    assert np.allclose(np.abs(np.angle(vis_bin)),
                       np.abs(np.angle(interpolated_bin)), atol=1e-2)
