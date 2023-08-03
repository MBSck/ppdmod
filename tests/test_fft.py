import astropy.units as u
import numpy as np
import pytest

from ppdmod.fft import compute_2Dfourier_transform, interpolate_for_coordinates
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


@pytest.mark.parametrize(
    "diameter, dim",
    [tuple([diameter, dim])
     for dim in [1024, 2048, 4096]
     for diameter in [4, 10, 20]*u.mas])
def test_interpolation(diameter: u.mas, dim: float,
                       ucoord: u.m, vcoord: u.m) -> None:
    """This tests the interpolation of the Fourier transform,
    but more importantly, implicitly the unit conversion of the
    frequency axis."""
    pixel_size, wavelength = 0.1*u.mas, 1.02322101e-05*u.m
    ft = compute_2Dfourier_transform(uniform_disk(pixel_size, dim,
                                                  diameter=diameter))
    vis = uniform_disk_vis(diameter, ucoord, vcoord, wavelength)
    interpolated_values = interpolate_for_coordinates(ft, dim,
                                                      pixel_size.to(u.rad),
                                                      ucoord, vcoord,
                                                      wavelength)
    interpolated_values /= ft[dim//2, dim//2]
    interpolated_values = np.real(interpolated_values)
    assert np.allclose(vis, interpolated_values, atol=1e-2)
