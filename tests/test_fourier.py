import pytest

import numpy as np
import astropy.units as u
import astropy.constants as c

from ppdmod.functionality.utils import make_ring_component, temperature_gradient,\
    optical_depth_gradient, flux_per_pixel
from ppdmod.model_components import RingComponent
from ppdmod.functionality.fourier import FastFourierTransform

################################### Fixtures #############################################

@pytest.fixture
def mock_wavelength():
    return 8*u.um

@pytest.fixture
def mock_pixel_scaling():
    return 50*u.mas/u.Quantity(128, unit=u.dimensionless_unscaled, dtype=int)

@pytest.fixture
def mock_zero_padding_two():
    return 2

@pytest.fixture
def mock_init_values():
    """The sublimation temp [astropy.units.K], the eff_temp [astropy.units.K], the
    distance of the star [astropy.units.pc], the luminosity [astropy.units.L_sun] and
    the wavelength [u.um]"""
    image_size = u.Quantity(128, unit=u.dimensionless_unscaled, dtype=int)
    return [50*u.mas, image_size, 1500*u.K,
            7900*u.K, 140*u.pc, 19*c.L_sun, image_size]

@pytest.fixture
def mock_flux_image(mock_wavelength, mock_init_values):
    ring_component = make_ring_component("inner_ring",
                                         [[0., 1.], [0, 180], [3., 5.], [0., 0.]])
    inner_radius = ring_component.params.inner_radius
    pixel_scaling = mock_init_values[0]/mock_init_values[1]
    ring = RingComponent(*mock_init_values)
    image = ring.eval_model(ring_component.params)

    temperature = temperature_gradient(image, 0.55, inner_radius, mock_init_values[2])

    optical_depth = optical_depth_gradient(image, 0.55, inner_radius, 1)
    return flux_per_pixel(mock_wavelength, temperature, optical_depth, pixel_scaling)

@pytest.fixture
def mock_init_values_one(mock_flux_image, mock_wavelength, mock_pixel_scaling):
    return mock_flux_image, mock_wavelength, mock_pixel_scaling

@pytest.fixture
def mock_init_values_two(mock_flux_image, mock_wavelength,
                         mock_pixel_scaling, mock_zero_padding_two):
    return mock_flux_image, mock_wavelength, mock_pixel_scaling, mock_zero_padding_two

################################ FOURIER - TESTS #########################################

def test_init(mock_init_values_one, mock_flux_image):
    fourier = FastFourierTransform(*mock_init_values_one)
    assert fourier.model_unpadded_dim == mock_flux_image.shape[0]
    assert fourier.model_unpadded_centre == mock_flux_image.shape[0]//2

def test_model_shape_properties(mock_init_values_one, mock_flux_image):
    fourier = FastFourierTransform(*mock_init_values_one)
    assert fourier.model_shape == mock_flux_image.shape
    assert fourier.dim == mock_flux_image.shape[0]
    assert fourier.model_centre == mock_flux_image.shape[0]//2

def test_frequ_axis_property(mock_init_values_one):
    fourier = FastFourierTransform(*mock_init_values_one)
    assert fourier.freq_axis.unit == 1/u.mas

def test_fftscling2m_property(mock_init_values_one):
    fourier = FastFourierTransform(*mock_init_values_one)
    assert fourier.fftscaling2m.unit == u.m

def test_fftscaling2Mlambda_property(mock_init_values_one):
    fourier = FastFourierTransform(*mock_init_values_one)
    assert fourier.fftaxis_Mlambda.unit == u.dimensionless_unscaled

def test_fftaxis_m_end_property(mock_init_values_one):
    fourier = FastFourierTransform(*mock_init_values_one)
    assert fourier.fftaxis_m_end.unit == u.m

def test_fftaxis_Mlambda_end_property(mock_init_values_one):
    fourier = FastFourierTransform(*mock_init_values_one)
    assert fourier.fftaxis_Mlambda_end.unit == u.dimensionless_unscaled

def test_fftaxis_m_property(mock_init_values_one):
    fourier = FastFourierTransform(*mock_init_values_one)
    assert fourier.fftaxis_m.unit == u.m
    assert isinstance(fourier.fftaxis_m.value, np.ndarray)
    assert fourier.fftaxis_m.shape[0] == fourier.dim

def test_fftaxis_Mlambda_property(mock_init_values_one):
    fourier = FastFourierTransform(*mock_init_values_one)
    assert fourier.fftaxis_Mlambda.unit == u.dimensionless_unscaled
    assert isinstance(fourier.fftaxis_Mlambda.value, np.ndarray)
    assert fourier.fftaxis_Mlambda.shape[0] == fourier.dim

def test_grid_m(mock_init_values_one):
    fourier = FastFourierTransform(*mock_init_values_one)
    assert isinstance(fourier.grid_m, list)
    assert isinstance(fourier.grid_m[0], u.Quantity)
    assert isinstance(fourier.grid_m[0].value, np.ndarray)


def test_zero_padding(mock_init_values_one, mock_init_values_two):
    fourier_zero_pad_one = FastFourierTransform(*mock_init_values_one)
    fourier_zero_pad_two = FastFourierTransform(*mock_init_values_two)
    assert fourier_zero_pad_one.dim == 2**7
    assert fourier_zero_pad_one.zero_padding == 2**8
    assert fourier_zero_pad_two.zero_padding == 2**9

# TODO: Implement test
def test_zero_pad_model(mock_init_values_one, mock_init_values_two):
    fourier_zero_pad_one = FastFourierTransform(*mock_init_values_one)
    fourier_zero_pad_two = FastFourierTransform(*mock_init_values_two)

# TODO: Implement test
def test_get_uv2fft2():
    ...

# TODO: Implement test
def test_get_amp_phase():
    ...

# TODO: Implement test
def do_fft2():
    ...

# NOTE: This has no return as it is only a plotting function
def test_plot_amp_phase():
    ...
