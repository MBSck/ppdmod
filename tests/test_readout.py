import pytest
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

def test_get_vis(example_fits_file_path):
    readout = ReadoutFits(example_fits_file_path)
    vis, viserr, sta_index = readout.get_vis()

    assert isinstance(vis, np.ndarray)
    assert isinstance(viserr, np.ndarray)
    assert vis.shape == (6, 121)
    assert viserr.shape == (6, 121)
    assert sta_index.shape == (6, 2)

    if np.max(vis) >= 1.:
        assert sta_index[0][0].unit == u.dimensionless_unscaled
