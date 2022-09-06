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

def test_get_data_for_wavelength(example_fits_file_path):
    readout = ReadoutFits(example_fits_file_path)
    visdata = readout.get_visibilities()
    wl_ind = np.array([random.randint(0, 120)])
    wl_indices = np.array([random.randint(0, 120) for _ in range(5)])
    len_wl_indices = len(wl_indices)
    vis4wl_singular, viserr4wl_singular = readout.get_data_for_wavelength(visdata, wl_ind)
    vis4wl, viserr4wl = readout.get_data_for_wavelength(visdata, wl_indices)
    assert isinstance(vis4wl_singular.value, np.ndarray)
    assert isinstance(viserr4wl_singular.value, np.ndarray)
    assert isinstance(vis4wl.value, np.ndarray)
    assert isinstance(viserr4wl.value, np.ndarray)
    assert vis4wl_singular.shape == (1, 6)
    assert viserr4wl_singular.shape == (1, 6)
    assert vis4wl.shape == (len_wl_indices, 6)
    assert viserr4wl.shape == (len_wl_indices, 6)
    assert vis4wl_singular.unit == u.Jy
    assert viserr4wl_singular.unit == u.Jy
    assert vis4wl.unit == u.Jy
    assert viserr4wl.unit == u.Jy

def test_get_telescope_information(example_fits_file_path):
    readout = ReadoutFits(example_fits_file_path)
    station_names, station_indicies,\
            station_indicies4baselines,\
            station_indicies4triangles = readout.get_telescope_information()
    assert isinstance(station_names, np.ndarray)
    assert isinstance(station_names[0], str)
    assert isinstance(station_indicies.value, np.ndarray)
    assert isinstance(station_indicies4baselines.value, np.ndarray)
    assert isinstance(station_indicies4triangles.value, np.ndarray)
    assert station_indicies.unit == u.dimensionless_unscaled

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

def test_get_visibilities4wavelength(example_fits_file_path):
    readout = ReadoutFits(example_fits_file_path)
    wl_ind = np.array([random.randint(0, 120)])
    wl_indices = np.array([random.randint(0, 120) for _ in range(5)])
    len_wl_indices = len(wl_indices)
    vis_singular, viserr_singular = readout.get_flux4wavelength(wl_ind)
    vis, viserr = readout.get_flux4wavelength(wl_indices)
    assert isinstance(vis_singular.value, np.ndarray)
    assert isinstance(viserr_singular.value, np.ndarray)
    assert isinstance(vis.value, np.ndarray)
    assert isinstance(vis.value, np.ndarray)
    assert vis_singular.shape == (1, 6)
    assert viserr_singular.shape == (1, 6)
    assert vis4wl.shape == (len_wl_indices, 6)
    assert viserr4wl.shape == (len_wl_indices, 6)

    if np.max(vis.value) >= 1.:
        assert vis_singular.unit == u.Jy
        assert viserr_singular.unit == u.Jy
        assert vis.unit == u.Jy
        assert viserr.unit == u.Jy
    else:
        assert vis_singular.unit == u.dimensionless_unscaled
        assert viserr_singular.unit == u.dimensionless_unscaled
        assert vis.unit == u.dimensionless_unscaled
        assert viserr.unit == u.dimensionless_unscaled

def test_get_flux4wavlength(example_fits_file_path):
    readout = ReadoutFits(example_fits_file_path)
    wl_ind = np.array([random.randint(0, 120)])
    wl_indices = np.array([random.randint(0, 120) for _ in range(5)])
    len_wl_indices = len(wl_indices)
    flux4wl_singular, fluxerr4wl_singular = readout.get_flux4wavelength(wl_ind)
    flux4wl, fluxerr4wl = readout.get_flux4wavelength(wl_indices)
    assert isinstance(flux4wl_singular.value, np.ndarray)
    assert isinstance(fluxerr4wl_singular.value, np.ndarray)
    assert isinstance(flux4wl.value, np.ndarray)
    assert isinstance(fluxerr4wl.value, np.ndarray)
    assert flux4wl_singular.value.shape == (1, )
    assert fluxerr4wl_singular.value.shape == (1, )
    assert flux4wl.value.shape == (len_wl_indices, )
    assert fluxerr4wl.value.shape == (len_wl_indices, )
    assert flux4wl_singular.unit == u.Jy
    assert fluxerr4wl_singular.unit == u.Jy
    assert flux4wl.unit == u.Jy
    assert fluxerr4wl.unit == u.Jy

def test_telescope_information_from_different_header(example_fits_file_path):
    readout = ReadoutFits(example_fits_file_path)
    station_names, station_indices,\
            station_indices4baselines,\
            station_indices4triangles = readout.get_telescope_information()
    station_indices_from_visibilities = readout.get_data("oi_vis", "sta_index")[0]
    station_indices_from_visibilities_squared = readout.\
            get_data("oi_vis2", "sta_index")[0]

    assert np.all(station_indices_from_visibilities ==\
            station_indices_from_visibilities_squared)
