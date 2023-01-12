import pytest
import random

from ppdmod.lib.data_prep import DataHandler

################################### Fixtures #############################################

@pytest.fixture
def mock_fits_file():
    """This is an N-band test (.fits)-file"""
    return "../data/tests/test.fits"

@pytest.fixture
def mock_two_fits_files(mock_fits_file):
    return [mock_fits_file]*2

@pytest.fixture
def selected_wavelengths():
    return [8.5]*u.um, [8.5, 10.0]*u.um, [8.5]*random.randint(1, 10)*u.um

################################ DataDandler - TESTS #####################################

def test_data_handler_init(mock_fits_file, mock_two_fits_files, selected_wavelengths):
    selected_wl_solo, selected_wl, _ = selected_wavelengths
    # Note: This also tests the initalisation with one or more wavelengths
    data_handler_two = DataHandler(mock_fits_file, selected_wl_solo)
    data_handler_three = DataHandler(mock_two_fits_files, selected_wl)
    assert len(data_handler_two.readouts) == 1
    assert len(data_handler_three.readouts) == 2

def test_get_data_type_function(mock_fits_file, mock_two_fits_files, selected_wavelengths):
    selected_wl_solo, selected_wl, _ = selected_wavelengths
    data_handler = DataHandler(mock_two_fits_files, selected_wl)
    readout = data_handler.readouts[0]
    assert readout.get_visibilities4wavelength ==\
        data_handler._get_data_type_function(readout, "vis")
    assert readout.get_visibilities_squared4wavelength ==\
        data_handler._get_data_type_function(readout, "vis2")
    assert readout.get_closure_phases4wavelength ==\
        data_handler._get_data_type_function(readout, "cphases")
    assert readout.get_flux4wavelength ==\
        data_handler._get_data_type_function(readout, "flux")

def test_iterate_over_data_arrays(mock_fits_file, mock_two_fits_files, selected_wavelengths):
    selected_wl_solo, selected_wl, selected_wl_multi = selected_wavelengths
    len_selected_wl, len_selected_wl_multi = len(selected_wl), len(selected_wl_multi)
    # Note: This also tests the initalisation with one or more wavelengths
    data_handler = DataHandler(two_fits_files, selected_wl_solo)
    data_handler_two = DataHandler(two_fits_files, selected_wl)
    data_handler_two_multi = DataHandler(two_fits_files, selected_wl_multi)

    visdata_solo = data_handler_two_solo.readout_files[0].\
        get_visibilities4wavelength(data_handler_two_solo.wl_ind)
    visdata = data_handler_two.readouts[0].\
        get_visibilities4wavelength(data_handler_two.wl_ind)
    visdata_multi = data_handler_two_multi.readouts[0].\
        get_visibilities4wavelength(data_handler_two_multi.wl_ind)
    merged_data_vis_solo = data_handler_two_solo.\
        _iterate_over_data_arrays(visdata_solo, visdata_solo.copy())
    merged_data_vis = data_handler_two._iterate_over_data_arrays(visdata, visdata.copy())
    merged_data_vis_multi = data_handler_two_multi.\
        _iterate_over_data_arrays(visdata_multi, visdata_multi.copy())

    cphasesdata_solo = data_handler_two_solo.readout_files[0].\
        get_closure_phases4wavelength(data_handler_two_solo.wl_ind)
    cphasesdata = data_handler_two.readouts[0].\
        get_closure_phases4wavelength(data_handler_two.wl_ind)
    cphasesdata_multi = data_handler_two_multi.readouts[0].\
        get_closure_phases4wavelength(data_handler_two_multi.wl_ind)
    merged_data_cphases_solo = data_handler_two_solo.\
        _iterate_over_data_arrays(cphasesdata_solo, cphasesdata_solo.copy())
    merged_data_cphases = data_handler_two.\
        _iterate_over_data_arrays(cphasesdata, cphasesdata.copy())
    merged_data_cphases_multi = data_handler_two_multi.\
        _iterate_over_data_arrays(cphasesdata_multi, cphasesdata_multi.copy())

    fluxdata_solo = data_handler_two_solo.readout_files[0].\
        get_flux4wavelength(data_handler_two_solo.wl_ind)
    fluxdata = data_handler_two.readouts[0].\
        get_flux4wavelength(data_handler_two.wl_ind)
    fluxdata_multi = data_handler_two_multi.readouts[0].\
        get_flux4wavelength(data_handler_two_multi.wl_ind)
    merged_data_fluxdata_solo = data_handler_two_solo.\
        _iterate_over_data_arrays(fluxdata_solo, fluxdata_solo.copy())
    merged_data_fluxdata = data_handler_two.\
        _iterate_over_data_arrays(fluxdata, fluxdata.copy())
    merged_data_fluxdata_multi = data_handler_two_multi.\
        _iterate_over_data_arrays(fluxdata_multi, fluxdata_multi.copy())

    assert merged_data_vis_solo[0].shape == (1, 12)
    assert merged_data_vis_solo[1].shape == (1, 12)
    assert merged_data_vis[0].shape == (len_selected_wl, 12)
    assert merged_data_vis[1].shape == (len_selected_wl, 12)
    assert merged_data_vis_multi[0].shape == (len_selected_wl_multi, 12)
    assert merged_data_vis_multi[1].shape == (len_selected_wl_multi, 12)

    assert merged_data_cphases_solo[0].shape == (1, 8)
    assert merged_data_cphases_solo[1].shape == (1, 8)
    assert merged_data_cphases[0].shape == (len_selected_wl, 8)
    assert merged_data_cphases[1].shape == (len_selected_wl, 8)
    assert merged_data_cphases_multi[0].shape == (len_selected_wl_multi, 8)
    assert merged_data_cphases_multi[1].shape == (len_selected_wl_multi, 8)

    assert merged_data_fluxdata_solo[0].shape == (1, 2)
    assert merged_data_fluxdata_solo[1].shape == (1, 2)
    assert merged_data_fluxdata[0].shape == (len_selected_wl, 2)
    assert merged_data_fluxdata[1].shape == (len_selected_wl, 2)
    assert merged_data_fluxdata_multi[0].shape == (len_selected_wl_multi, 2)
    assert merged_data_fluxdata_multi[1].shape == (len_selected_wl_multi, 2)

def test_merge_data(example_fits_files_lists, selected_wavelengths):
    selected_wl_solo, selected_wl, selected_wl_multi = selected_wavelengths
    len_selected_wl, len_selected_wl_multi = len(selected_wl), len(selected_wl_multi)
    two_fits_files, three_fits_files = example_fits_files_lists
    data_handler_two_solo = DataHandler(two_fits_files, selected_wl_solo)
    data_handler_two = DataHandler(two_fits_files, selected_wl)
    data_handler_two_multi = DataHandler(two_fits_files, selected_wl_multi)
    data_handler_three_solo = DataHandler(three_fits_files, selected_wl_solo)
    data_handler_three = DataHandler(three_fits_files, selected_wl)
    data_handler_three_multi = DataHandler(three_fits_files, selected_wl_multi)
    data_handler_three_multi = DataHandler(three_fits_files, selected_wl_multi)

    assert data_handler_two_solo._merge_data("vis")[0].shape == (1, 12)
    assert data_handler_two._merge_data("vis")[0].shape == (len_selected_wl, 12)
    assert data_handler_two_multi._merge_data("vis")[0].shape ==\
        (len_selected_wl_multi, 12)
    assert data_handler_three_solo._merge_data("vis")[0].shape == (1, 18)
    assert data_handler_three._merge_data("vis")[0].shape == (len_selected_wl, 18)
    assert data_handler_three_multi._merge_data("vis")[0].shape ==\
        (len_selected_wl_multi, 18)

    assert data_handler_two_solo._merge_data("cphases")[0].shape == (1, 8)
    assert data_handler_two._merge_data("cphases")[0].shape == (len_selected_wl, 8)
    assert data_handler_two_multi._merge_data("cphases")[0].shape ==\
        (len_selected_wl_multi, 8)
    assert data_handler_three_solo._merge_data("cphases")[0].shape == (1, 12)
    assert data_handler_three._merge_data("cphases")[0].shape == (len_selected_wl, 12)
    assert data_handler_three_multi._merge_data("cphases")[0].shape ==\
        (len_selected_wl_multi, 12)

