from pathlib import Path

import astropy.units as u
import numpy as np
import pytest

from ppdmod.data import ReadoutFits, set_fit_wavelengths, get_data
from ppdmod.options import OPTIONS


@pytest.fixture
def fits_file() -> Path:
    """A MATISSE (.fits)-file."""
    return Path("data/fits") /\
        "hd_142666_2022-04-23T03_05_25:2022-04-23T02_28_06_AQUARIUS_FINAL_TARGET_INT.fits"


@pytest.fixture
def readout() -> ReadoutFits:
    """The data of the input files."""
    file = "hd_142666_2022-04-23T03_05_25:2022-04-23T02_28_06_AQUARIUS_FINAL_TARGET_INT.fits"
    return ReadoutFits(Path("data/fits") / file)


@pytest.fixture
def wavelength() -> u.m:
    """A wavelength grid."""
    return (8.28835527e-06*u.m).to(u.um)


@pytest.fixture
def wavelengths() -> u.um:
    """A wavelength grid."""
    return ([8.28835527e-06, 1.02322101e-05]*u.m).to(u.um)


def test_readout_init(readout: ReadoutFits) -> None:
    """Tests the readout of the (.fits)-files."""
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
    "wavelength", [(8.28835527e-06*u.m).to(u.um),
                   ([8.28835527e-06, 1.02322101e-05]*u.m).to(u.um)])
def test_wavelength_retrieval(readout: ReadoutFits, wavelength: u.um) -> None:
    """Tests the wavelength retrieval of the (.fits)-files."""
    total_flux = readout.get_data_for_wavelengths(wavelength, key="flux")
    total_flux_err = readout.get_data_for_wavelengths(wavelength, key="flux_err")
    corr_flux = readout.get_data_for_wavelengths(wavelength, key="vis")
    corr_flux_err = readout.get_data_for_wavelengths(wavelength, key="vis_err")
    cphase = readout.get_data_for_wavelengths(wavelength, key="t3phi")
    cphase_err = readout.get_data_for_wavelengths(wavelength, key="t3phi_err")
    assert total_flux and total_flux_err
    assert corr_flux and corr_flux_err
    assert cphase and cphase_err
    assert all(flx.shape == () for flx in total_flux.values())
    assert all(flx_err.shape == () for flx_err in total_flux_err.values())
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
        set_fit_wavelengths(*wavelength.to(u.m).value)
    assert np.array_equal(OPTIONS["fit.wavelengths"], wavelength)

    set_fit_wavelengths()
    assert not OPTIONS["fit.wavelengths"]


@pytest.mark.parametrize(
    "wavelength", [([8.28835527e-06]*u.m).to(u.um),
                   ([8.28835527e-06, 1.02322101e-05]*u.m).to(u.um)])
def test_get_data(fits_file: Path, wavelength: u.um) -> None:
    """Tests the automatic data procurrment from one
    or multiple (.fits)-files."""
    set_fit_wavelengths(*wavelength)
    get_data(fits_file)
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
    assert all(all(flx.shape == () for flx in flux.values())
               for flux in total_flux)
    assert all(all(flxe.shape == () for flxe in flux_err.values())
               for flux_err in total_flux_err)
    assert all(all(cflx.shape == (6,) for cflx in corr_flux.values())
               for corr_flux in corr_flux)
    assert all(all(cflxe.shape == (6,) for cflxe in corr_flux_err.values())
               for corr_flux_err in corr_flux_err)
    assert all(all(c.shape == (4,) for c in cp.values()) for cp in cphase)
    assert all(all(ce.shape == (4,) for ce in cp_err.values())
               for cp_err in cphase_err)

    get_data()
    assert not OPTIONS["data.total_flux"]
    assert not OPTIONS["data.total_flux_error"]
    assert not OPTIONS["data.correlated_flux"]
    assert not OPTIONS["data.correlated_flux_error"]
    assert not OPTIONS["data.closure_phase"]
    assert not OPTIONS["data.closure_phase_error"]
