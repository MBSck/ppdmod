import pytest
import random
import numpy as np
import astropy.units as u

from astropy.io import fits
from collections import namedtuple

from ppdmod.functionality.readout import ReadoutFits

@pytest.fixture
def example_fits_file_path():
    return "../data/test.fits"

@pytest.fixture
def header_names_tuple():
    Data = namedtuple("Data", ["header", "data", "error", "station"])
    vis = Data("oi_vis", "visamp", "visamperr", "sta_index")
    vis2 = Data("oi_vis2", "vis2amp", "vis2amperr", "sta_index")
    t3phi = Data("oi_t3phi", "t3phi", "t3phierr", "sta_index")
    flux = Data("oi_flux", "fluxdata", "fluxerr", None)
    wavelength = Data("oi_wavelength", "eff_wave", None, None)
    Header = namedtuple("Header", ["vis", "vis2", "cphase", "flux", "wavelength"])
    return Header(vis, vis2, t3phi, flux, wavelength)

# TODO: Implement this test
def test_get_info():
    ...

# TODO: Implement this test
def test_get_header():
    ...

def test_get_data(example_fits_file_path, header_names_tuple):
    """Tests if all MATISSE values can be fetched from the (.fits)-file"""
    readout = ReadoutFits(example_fits_file_path)
    output =  readout.get_data(header_names_tuple.vis.header,
            header_names_tuple.vis.data, header_names_tuple.vis.error,
            header_names_tuple.vis.station)

    data, error, sta_index = output

    with fits.open(example_fits_file_path) as hdul:
        data_fits = hdul[header_names_tuple.vis.header].data[header_names_tuple.vis.data]
        error_fits = hdul[header_names_tuple.vis.header].data[header_names_tuple.vis.error]
        sta_index_fits = hdul[header_names_tuple.vis.header]\
                .data[header_names_tuple.vis.station]

    assert len(output) == 3
    assert isinstance(data, np.ndarray)
    assert isinstance(error, np.ndarray)
    assert isinstance(sta_index, np.ndarray)
    assert np.all(data == data_fits)
    assert np.all(error == error_fits)
    assert np.all(sta_index == sta_index_fits)

def test_get_telescope_information(example_fits_file_path):
    readout = ReadoutFits(example_fits_file_path)
    station_names, station_indicies,\
            station_indicies4baselines,\
            station_indicies4triangles = readout.get_telescope_information()
    assert isinstance(station_names, np.ndarray)
    assert isinstance(station_indicies.value, np.ndarray)
    assert isinstance(station_indicies4baselines.value, np.ndarray)
    assert isinstance(station_indicies4triangles.value, np.ndarray)
    assert station_indicies.unit == u.dimensionless_unscaled
    assert isinstance(station_names[0], str)

# TODO: Implement this test
def test_get_split_uvcoords():
    ...

# TODO: Implement this test
def test_get_uvcoords():
    ...

# TODO: Implement this test
def test_get_closure_phases_uvcoords():
    ...

def test_get_baselines(example_fits_file_path):
    readout = ReadoutFits(example_fits_file_path)
    baselines = readout.get_baselines()
    assert isinstance(baselines.value, np.ndarray)
    assert baselines.unit == u.m

def test_get_visibilities(example_fits_file_path):
    readout = ReadoutFits(example_fits_file_path)
    vis, viserr = readout.get_visibilities()

    assert isinstance(vis.value, np.ndarray)
    assert isinstance(viserr.value, np.ndarray)
    assert isinstance(vis.value[0], np.ndarray)
    assert isinstance(vis.value[0][0], float)
    assert isinstance(viserr.value[0], np.ndarray)
    assert isinstance(viserr.value[0][0], float)
    assert vis.value.shape == (6, 121)
    assert viserr.value.shape == (6, 121)

    if np.max(vis.value) >= 1.:
        assert vis.unit == u.Jy
        assert viserr.unit == u.Jy
    else:
        assert vis.unit == u.dimensionless_unscaled
        assert viserr.unit == u.dimensionless_unscaled

def test_get_visibilities_squared(example_fits_file_path):
    readout = ReadoutFits(example_fits_file_path)
    vis2, vis2err = readout.get_visibilities_squared()

    assert isinstance(vis2.value, np.ndarray)
    assert isinstance(vis2err.value, np.ndarray)
    assert isinstance(vis2.value[0], np.ndarray)
    assert isinstance(vis2.value[0][0], float)
    assert isinstance(vis2err.value[0], np.ndarray)
    assert isinstance(vis2err.value[0][0], float)
    assert vis2.value.shape == (6, 121)
    assert vis2err.value.shape == (6, 121)
    assert vis2.unit == u.dimensionless_unscaled
    assert vis2err.unit == u.dimensionless_unscaled

def test_get_closure_phases(example_fits_file_path):
    readout = ReadoutFits(example_fits_file_path)
    cphases, cphaseserr = readout.get_closure_phases()

    assert isinstance(cphases.value, np.ndarray)
    assert isinstance(cphaseserr.value, np.ndarray)
    assert isinstance(cphases.value[0], np.ndarray)
    assert isinstance(cphases.value[0][0], float)
    assert isinstance(cphaseserr.value[0], np.ndarray)
    assert isinstance(cphaseserr.value[0][0], float)
    assert cphases.value.shape == (4, 121)
    assert cphaseserr.value.shape == (4, 121)
    assert cphases.unit == u.deg
    assert cphaseserr.unit == u.deg

def test_get_flux(example_fits_file_path):
    readout = ReadoutFits(example_fits_file_path)
    flux, fluxerr = readout.get_flux()
    assert isinstance(flux.value, np.ndarray)
    assert isinstance(fluxerr.value, np.ndarray)
    assert flux.value.shape == (1, 121)
    assert fluxerr.value.shape == (1, 121)
    assert flux.unit== u.Jy
    assert fluxerr.unit== u.Jy

def test_get_wavelength_solution(example_fits_file_path):
    readout = ReadoutFits(example_fits_file_path)
    wavelength_solution = readout.get_wavelength_solution()
    assert isinstance(wavelength_solution.value, np.ndarray)
    assert wavelength_solution.unit == u.um

def test_get_flux4wavlength(example_fits_file_path):
    readout = ReadoutFits(example_fits_file_path)
    wavelength_index = np.array([random.randint(0, 120)])
    wavelength_indicies = np.array([random.randint(0, 120) for _ in range(5)])
    number_wavelength_indicies = len(wavelength_indicies)
    flux4wavelength_singular,\
            fluxerr4wavelength_singular = readout.get_flux4wavelength(wavelength_index)
    flux4wavelength, fluxerr4wavelength = readout.get_flux4wavelength(wavelength_indicies)
    assert isinstance(flux4wavelength_singular.value, np.ndarray)
    assert isinstance(fluxerr4wavelength_singular.value, np.ndarray)
    assert isinstance(flux4wavelength.value, np.ndarray)
    assert isinstance(fluxerr4wavelength.value, np.ndarray)
    assert flux4wavelength_singular.value.shape == (1, )
    assert fluxerr4wavelength_singular.value.shape == (1, )
    assert flux4wavelength.value.shape == (number_wavelength_indicies, )
    assert fluxerr4wavelength.value.shape == (number_wavelength_indicies, )
    assert flux4wavelength_singular.unit == u.Jy
    assert fluxerr4wavelength_singular.unit == u.Jy
    assert flux4wavelength.unit == u.Jy
    assert fluxerr4wavelength.unit == u.Jy

def test_telescope_information_from_different_header(example_fits_file_path):
    readout = ReadoutFits(example_fits_file_path)
    station_names, station_indicies,\
            station_indicies4baselines,\
            station_indicies4triangles = readout.get_telescope_information()
    station_indicies_from_visibilities = readout.get_data("oi_vis", "sta_index")[0]
    station_indicies_from_visibilities_squared = readout.get_data("oi_vis2", "sta_index")[0]

    assert np.all(station_indicies_from_visibilities ==\
            station_indicies_from_visibilities_squared)
