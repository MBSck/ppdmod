import astropy.units as u
import numpy as np
import pytest

from ppdmod.fft import compute_2Dfourier_transform, get_frequency_axis,\
    interpolate_coordinates
from ppdmod.utils import uniform_disk, uniform_disk_vis


# TODO: Make tests for the phase that make pictures.
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
def wavelength() -> u.um:
    """Sets the wavelength."""
    return (1.02322101e-05*u.m).to(u.um)


def test_compute2Dfourier_transform(pixel_size: u.mas) -> None:
    """Tests the computation of the 2D fourier transform."""
    ud = uniform_disk(pixel_size, 512, diameter=4*u.mas)
    ft = compute_2Dfourier_transform(ud)
    assert ft.shape == ud.shape
    assert ft.dtype == np.complex128


# TODO: Test input for both wavelength in meter and um?
# TODO: Check that the Parameter and such actually gets the right values (even with
# the new scheme). Also the set data.
def test_get_frequency_axis(pixel_size: u.mas, wavelength: u.um) -> None:
    """Tests the frequency axis calculation and transformation."""
    frequency_axis = get_frequency_axis(512, pixel_size, wavelength)
    assert frequency_axis.unit == u.m
    assert frequency_axis.shape == (512, )


@pytest.mark.parametrize(
    "diameter, dim",
    [tuple([diameter, dim])
     for dim in [1024, 2048, 4096]
     for diameter in [4, 10, 20]*u.mas])
def test_cphases_interpolation(diameter: u.mas, dim: float,
                               u123coord: u.m, v123coord: u.m,
                               pixel_size: u.mas, wavelength: u.um) -> None:
    """Tests the interpolation of the closure phases."""
    ft = compute_2Dfourier_transform(uniform_disk(pixel_size, dim,
                                                  diameter=diameter))
    interpolated_cphase = interpolate_coordinates(
        ft, dim, pixel_size, u123coord, v123coord, wavelength)
    interpolated_cphase = np.product(
        interpolated_cphase/ft[dim//2, dim//2], axis=1)
    interpolated_cphase = np.real(interpolated_cphase)

    cphase = []
    for ucoord, vcoord in zip(u123coord, v123coord):
        tmp_cphase = uniform_disk_vis(diameter, ucoord, vcoord, wavelength)
        cphase.append(tmp_cphase)
    cphase = np.product(cphase, axis=0)
    if dim == 1024:
        assert np.allclose(cphase, interpolated_cphase, atol=1e-1)
    else:
        assert np.allclose(cphase, interpolated_cphase, atol=1e-2)


@pytest.mark.parametrize(
    "diameter, dim",
    [tuple([diameter, dim])
     for dim in [1024, 2048, 4096]
     for diameter in [4, 10, 20]*u.mas])
def test_vis_interpolation(diameter: u.mas, dim: float,
                           ucoord: u.m, vcoord: u.m,
                           pixel_size: u.mas, wavelength: u.um) -> None:
    """This tests the interpolation of the Fourier transform,
    but more importantly, implicitly the unit conversion of the
    frequency axis for the visibilitites/correlated fluxes."""
    ft = compute_2Dfourier_transform(uniform_disk(pixel_size, dim,
                                                  diameter=diameter))
    vis = uniform_disk_vis(diameter, ucoord, vcoord, wavelength)
    interpolated_values = interpolate_coordinates(ft, dim,
                                                  pixel_size,
                                                  ucoord, vcoord,
                                                  wavelength)
    interpolated_values /= ft[dim//2, dim//2]
    interpolated_values = np.real(interpolated_values)
    assert np.allclose(vis, interpolated_values, atol=1e-2)
