from typing import List
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest

from ppdmod.data import ReadoutFits, set_fit_wavelengths, set_data
from ppdmod.options import OPTIONS


@pytest.fixture
def fits_files() -> List[Path]:
    """A MATISSE (.fits)-file."""
    files = [
        "hd_142666_2022-04-23T03_05_25:2022-04-23T02_28_06_HAWAII-2RG_FINAL_TARGET_INT.fits",
        "hd_142666_2022-04-23T03_05_25:2022-04-23T02_28_06_AQUARIUS_FINAL_TARGET_INT.fits"]
    return [Path("data/fits") / file for file in files]


@pytest.fixture
def readouts(fits_files: Path) -> List[ReadoutFits]:
    """The data of the input files."""
    return [ReadoutFits(file) for file in fits_files]


@pytest.mark.parametrize(
    "fits_file",
    ["hd_142666_2022-04-23T03_05_25:2022-04-23T02_28_06_HAWAII-2RG_FINAL_TARGET_INT.fits",
     "hd_142666_2022-04-23T03_05_25:2022-04-23T02_28_06_AQUARIUS_FINAL_TARGET_INT.fits"]
)
def test_readout_init(fits_file: str) -> None:
    """Tests the readout of the (.fits)-files."""
    readout = ReadoutFits(Path("data/fits") / fits_file)
    assert readout.ucoord.shape == (6,)
    assert readout.vcoord.shape == (6,)
    assert readout.u1coord.shape == (4,)
    assert readout.u2coord.shape == (4,)
    assert readout.u3coord.shape == (4,)
    assert readout.v1coord.shape == (4,)
    assert readout.v2coord.shape == (4,)
    assert readout.v3coord.shape == (4,)
    assert len(readout.u123coord) == 3
    assert len(readout.v123coord) == 3
    assert readout.wavelength.unit == u.um


@pytest.mark.parametrize(
    "wavelength",
    [(8.28835527e-06*u.m).to(u.um),
     ([8.28835527e-06, 1.02322101e-05]*u.m).to(u.um)])
def test_wavelength_retrieval(readouts: List[ReadoutFits], wavelength: u.um) -> None:
    """Tests the wavelength retrieval of the (.fits)-files."""
    for index, readout in enumerate(readouts):
        total_flux = readout.get_data_for_wavelength(wavelength, key="flux")
        total_flux_err = readout.get_data_for_wavelength(wavelength, key="flux_err")
        corr_flux = readout.get_data_for_wavelength(wavelength, key="vis")
        corr_flux_err = readout.get_data_for_wavelength(wavelength, key="vis_err")
        cphase = readout.get_data_for_wavelength(wavelength, key="t3phi")
        cphase_err = readout.get_data_for_wavelength(wavelength, key="t3phi_err")
        if index == 0:
            assert not total_flux and not total_flux_err
            assert not corr_flux and not corr_flux_err
            assert not cphase and not cphase_err
        else:
            assert total_flux and total_flux_err
            assert corr_flux and corr_flux_err
            assert cphase and cphase_err
            assert all(flx.shape == (1,) for flx in total_flux.values())
            assert all(flx_err.shape == (1,) for flx_err in total_flux_err.values())
            assert all(cp.shape == (4,) for cp in cphase.values())
            assert all(cp_err.shape == (4,) for cp_err in cphase_err.values())
            assert all(corr_flx.shape == (6,) for corr_flx in corr_flux.values())
            assert all(corr_flx_err.shape == (6,) for corr_flx_err in corr_flux_err.values())
            assert isinstance(total_flux, dict) and isinstance(total_flux_err, dict)
            assert isinstance(corr_flux, dict) and isinstance(corr_flux_err, dict)
            assert isinstance(cphase, dict) and isinstance(cphase_err, dict)


@pytest.mark.parametrize(
    "wavelength", [([8.28835527e-06]*u.m).to(u.um),
                   ([8.28835527e-06, 1.02322101e-05]*u.m).to(u.um)])
def test_set_fit_wavelenghts(wavelength: u.um) -> None:
    """Tests the set fit wavelenghts function."""
    if wavelength.size == 1:
        set_fit_wavelengths(wavelength.to(u.m).value)
    else:
        set_fit_wavelengths(wavelength.to(u.m).value)
    assert np.array_equal(OPTIONS["fit.wavelengths"], wavelength)

    set_fit_wavelengths()
    assert not OPTIONS["fit.wavelengths"]


@pytest.mark.parametrize(
    "wavelength",
    [#([8.28835527e-06]*u.m).to(u.um),
     #([8.28835527e-06, 1.02322101e-05]*u.m).to(u.um),
     ([3.520375e-06, 8.28835527e-06, 1.02322101e-05]*u.m).to(u.um)])
def test_get_data(fits_files: Path, wavelength: u.um) -> None:
    """Tests the automatic data procurrment from one
    or multiple (.fits)-files."""
    set_fit_wavelengths(wavelength)
    set_data(fits_files)
    total_flux, total_flux_err =\
        OPTIONS["data.total_flux"], OPTIONS["data.total_flux_error"]
    corr_flux, corr_flux_err =\
        OPTIONS["data.correlated_flux"], OPTIONS["data.correlated_flux_error"]
    cphase, cphase_err =\
        OPTIONS["data.closure_phase"], OPTIONS["data.closure_phase_error"]

    assert isinstance(total_flux, list)
    assert isinstance(total_flux_err, list)
    assert isinstance(corr_flux, list)
    assert isinstance(corr_flux_err, list)
    assert isinstance(cphase, list)
    assert isinstance(cphase_err, list)
    assert all(all(flx.shape == (1,) for flx in flux.values())
               for flux in total_flux)
    assert all(all(flxe.shape == (1,) for flxe in flux_err.values())
               for flux_err in total_flux_err)
    assert all(all(cflx.shape == (6,) for cflx in corr_flux.values())
               for corr_flux in corr_flux)
    assert all(all(cflxe.shape == (6,) for cflxe in corr_flux_err.values())
               for corr_flux_err in corr_flux_err)
    assert all(all(c.shape == (4,) for c in cp.values()) for cp in cphase)
    assert all(all(ce.shape == (4,) for ce in cp_err.values())
               for cp_err in cphase_err)

    set_data()
    assert not OPTIONS["data.total_flux"]
    assert not OPTIONS["data.total_flux_error"]
    assert not OPTIONS["data.correlated_flux"]
    assert not OPTIONS["data.correlated_flux_error"]
    assert not OPTIONS["data.closure_phase"]
    assert not OPTIONS["data.closure_phase_error"]

    set_data(fits_files, wavelengths=wavelength)
    assert isinstance(total_flux, list)
    assert isinstance(total_flux_err, list)
    assert isinstance(corr_flux, list)
    assert isinstance(corr_flux_err, list)
    assert isinstance(cphase, list)
    assert isinstance(cphase_err, list)
    assert all(all(flx.shape == (1,) for flx in flux.values())
               for flux in total_flux)
    assert all(all(flxe.shape == (1,) for flxe in flux_err.values())
               for flux_err in total_flux_err)
    assert all(all(cflx.shape == (6,) for cflx in corr_flux.values())
               for corr_flux in corr_flux)
    assert all(all(cflxe.shape == (6,) for cflxe in corr_flux_err.values())
               for corr_flux_err in corr_flux_err)
    assert all(all(c.shape == (4,) for c in cp.values()) for cp in cphase)
    assert all(all(ce.shape == (4,) for ce in cp_err.values())
               for cp_err in cphase_err)
    set_data()
    set_fit_wavelengths()
