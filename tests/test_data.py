from typing import List
from types import SimpleNamespace
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest

from ppdmod.data import ReadoutFits, set_fit_wavelengths, set_data
from ppdmod.options import OPTIONS


@pytest.fixture
def fits_files() -> List[Path]:
    """A MATISSE (.fits)-file."""
    return list(Path("data/fits").glob("*2022-04-23*.fits"))


@pytest.fixture
def readouts(fits_files: Path) -> List[ReadoutFits]:
    """The data of the input files."""
    return list(map(ReadoutFits, fits_files))


def test_readout_init(fits_files: List[Path]) -> None:
    """Tests the readout of the (.fits)-files."""
    for fits_file in fits_files:
        readout = ReadoutFits(fits_file)
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
    [([8.28835527e-06]*u.m).to(u.um),
     ([8.28835527e-06, 1.02322101e-05]*u.m).to(u.um),
     ([3.520375e-06, 8.28835527e-06, 1.02322101e-05]*u.m).to(u.um)])
def test_set_fit_wavelenghts(wavelength: u.um) -> None:
    """Tests the set fit wavelenghts function."""
    if wavelength.size == 1:
        set_fit_wavelengths(wavelength.to(u.m).value)
    else:
        set_fit_wavelengths(wavelength.to(u.m).value)
    assert np.array_equal(OPTIONS["fit.wavelengths"], wavelength)

    set_fit_wavelengths()
    assert not OPTIONS["fit.wavelengths"]


# TODO: Test for all inputs of the set_data function
@pytest.mark.parametrize(
    "wavelength",
    [([8.28835527e-06]*u.m).to(u.um),
     ([8.28835527e-06, 1.02322101e-05]*u.m).to(u.um),
     ([3.520375e-06, 8.28835527e-06, 1.02322101e-05]*u.m).to(u.um)])
def test_get_data(fits_files: List[Path], wavelength: u.um) -> None:
    """Tests the automatic data procurrment from one
    or multiple (.fits)-files."""
    set_fit_wavelengths(wavelength)
    set_data(fits_files)

    for key in ["flux", "vis", "vis2", "cphase"]:
        data, data_err = map(lambda x: OPTIONS[f"data.{key}{x}"], ["", "_err"])
        if key == "flux":
            shape = (1,)
        elif key in ["vis", "vis2"]:
            shape = (6,)
        elif key == "t3phi":
            shape = (4,)

        assert OPTIONS[f"data.{key}"]
        assert OPTIONS[f"data.{key}_err"]
        assert isinstance(data, list)
        assert isinstance(data_err, list)
        assert all(isinstance(d, np.ndarray) for d in data)
        assert all(isinstance(d, np.ndarray) for d in data_err)

        assert all(d.shape == (wavelength.size, shape) for d in data)
        assert all(d.shape == (wavelength.size, shape) for d in data_err)

    set_data()
    for key in ["flux", "corr_flux", "cphase"]:
        assert not OPTIONS[f"data.{key}"]
        assert not OPTIONS[f"data.{key}_err"]

    set_data(fits_files, wavelengths=wavelength)
    set_data()
    set_fit_wavelengths()


@pytest.mark.parametrize(
    "wavelength",
    [(8.28835527e-06*u.m).to(u.um),
     ([8.28835527e-06, 1.02322101e-05]*u.m).to(u.um)])
def test_wavelength_retrieval(readouts: List[ReadoutFits], wavelength: u.um) -> None:
    """Tests the wavelength retrieval of the (.fits)-files."""
    for index, readout in enumerate(readouts):
        flux = readout.get_data_for_wavelength(wavelength, key="flux")
        flux_err = readout.get_data_for_wavelength(wavelength, key="flux_err")
        vis = readout.get_data_for_wavelength(wavelength, key="vis")
        vis_err = readout.get_data_for_wavelength(wavelength, key="vis_err")
        cphase = readout.get_data_for_wavelength(wavelength, key="t3phi")
        cphase_err = readout.get_data_for_wavelength(wavelength, key="t3phi_err")

        if index == 0:
            assert not flux and not flux_err
            assert not vis and not vis_err
            assert not cphase and not cphase_err
        else:
            assert flux and flux_err
            assert vis and vis_err
            assert cphase and cphase_err
            assert all(flx.shape == (1,) for flx in flux)
            assert all(flx_err.shape == (1,) for flx_err in flux_err)
            assert all(cp.shape == (4,) for cp in cphase)
            assert all(cp_err.shape == (4,) for cp_err in cphase_err)
            assert all(corr_flx.shape == (6,) for corr_flx in vis)
            assert all(corr_flx_err.shape == (6,) for corr_flx_err in vis_err)
            assert isinstance(flux, dict) and isinstance(flux_err, dict)
            assert isinstance(vis, dict) and isinstance(vis_err, dict)
            assert isinstance(cphase, dict) and isinstance(cphase_err, dict)
