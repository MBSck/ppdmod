from pathlib import Path
from typing import List

import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits

from ppdmod.data import ReadoutFits, set_data, set_fit_wavelengths
from ppdmod.options import OPTIONS


@pytest.fixture
def fits_files() -> List[Path]:
    """A MATISSE (.fits)-file."""
    data_dir = Path(__file__).parent.parent / "data" / "matisse"
    april_one_file = list(data_dir.glob("*2022-04-21*.fits"))
    april_two_files = list(data_dir.glob("*2022-04-23*.fits"))
    april_two_files.extend(april_one_file)
    return april_two_files


@pytest.fixture
def readouts(fits_files: List[Path]) -> List[ReadoutFits]:
    """The data of the input files."""
    return list(map(ReadoutFits, fits_files))


def test_read_into_namespace(
    fits_files: List[Path], readouts: List[ReadoutFits]
) -> None:
    """Tests the read_into_namespace function."""
    for fits_file, readout in zip(fits_files, readouts):
        with fits.open(fits_file) as hdul:
            instrument = hdul[0].header["instrume"].lower()
            sci_index = OPTIONS.data.gravity.index if instrument == "gravity" else None
            wl_index = 1 if instrument == "gravity" else None
            vis = hdul["oi_vis", sci_index]
            vis2 = hdul["oi_vis2", sci_index]
            t3 = hdul["oi_t3", sci_index]

            u1coord, u2coord = map(lambda x: t3.data[f"u{x}coord"], ["1", "2"])
            v1coord, v2coord = map(lambda x: t3.data[f"v{x}coord"], ["1", "2"])
            u3coord, v3coord = (u1coord + u2coord), (v1coord + v2coord)
            u123coord = np.array([u1coord, u2coord, u3coord])
            v123coord = np.array([v1coord, v2coord, v3coord])

            assert np.array_equal(readout.t3.value, t3.data["t3phi"][:, wl_index:])
            assert np.array_equal(readout.t3.err, t3.data["t3phierr"][:, wl_index:])
            assert np.array_equal(readout.t3.u123coord, u123coord)
            assert np.array_equal(readout.t3.v123coord, v123coord)

            assert np.array_equal(readout.vis.value, vis.data["visamp"][:, wl_index:])
            assert np.array_equal(readout.vis.err, vis.data["visamperr"][:, wl_index:])
            assert np.array_equal(readout.vis.ucoord, vis.data["ucoord"].reshape(1, -1))
            assert np.array_equal(readout.vis.vcoord, vis.data["vcoord"].reshape(1, -1))

            assert np.array_equal(
                readout.vis2.value, vis2.data["vis2data"][:, wl_index:]
            )
            assert np.array_equal(readout.vis2.err, vis2.data["vis2err"][:, wl_index:])
            assert np.array_equal(
                readout.vis2.ucoord, vis2.data["ucoord"].reshape(1, -1)
            )
            assert np.array_equal(
                readout.vis2.vcoord, vis2.data["vcoord"].reshape(1, -1)
            )

        assert readout.wavelength.unit == u.um
        wl_len = readout.wavelength.shape[0]

        assert readout.t3.value.shape == (4, wl_len)
        assert readout.t3.err.shape == (4, wl_len)
        assert readout.t3.u123coord.shape == (3, 4)
        assert readout.t3.v123coord.shape == (3, 4)

        assert readout.vis.value.shape == (6, wl_len)
        assert readout.vis.err.shape == (6, wl_len)
        assert readout.vis.ucoord.shape == (1, 6)
        assert readout.vis.vcoord.shape == (1, 6)

        assert readout.vis2.value.shape == (6, wl_len)
        assert readout.vis2.err.shape == (6, wl_len)
        assert readout.vis2.ucoord.shape == (1, 6)
        assert readout.vis2.vcoord.shape == (1, 6)


@pytest.mark.parametrize("wavelength", [[8], [8, 10], [3, 8, 10]])
def test_set_fit_wavelenghts(wavelength: u.um) -> None:
    """Tests the set fit wavelenghts function."""
    set_fit_wavelengths(wavelength)
    wavelength = [wavelength] if not isinstance(wavelength, list) else wavelength
    assert np.array_equal(OPTIONS.fit.wavelengths, wavelength * u.um)
    assert OPTIONS.fit.wavelengths.unit == u.um
    assert OPTIONS.fit.wavelengths.shape == (len(wavelength),)

    set_fit_wavelengths()
    assert not OPTIONS.fit.wavelengths


# TODO: Maybe add tests that show that nan is one of the results that can come out.
@pytest.mark.parametrize(
    "wavelength", [[3.5] * u.um, [3.5, 8] * u.um, [8] * u.um, [8, 10] * u.um]
)
def test_wavelength_retrieval(readouts: List[ReadoutFits], wavelength: u.um) -> None:
    """Tests the wavelength retrieval of the (.fits)-files."""
    wl_len = len(wavelength)
    for readout in readouts:
        flux = readout.get_data_for_wavelength(wavelength, "flux", "value")
        flux_err = readout.get_data_for_wavelength(wavelength, "flux", "err")
        t3 = readout.get_data_for_wavelength(wavelength, "t3", "value")
        t3_err = readout.get_data_for_wavelength(wavelength, "t3", "err")
        vis = readout.get_data_for_wavelength(wavelength, "vis", "value")
        vis_err = readout.get_data_for_wavelength(wavelength, "vis", "err")
        vis2 = readout.get_data_for_wavelength(wavelength, "vis2", "value")
        vis2_err = readout.get_data_for_wavelength(wavelength, "vis2", "err")

        assert isinstance(flux, np.ndarray) and isinstance(flux_err, np.ndarray)
        assert isinstance(vis, np.ndarray) and isinstance(vis_err, np.ndarray)
        assert isinstance(t3, np.ndarray) and isinstance(t3_err, np.ndarray)

        assert flux.size != 0 and flux_err.size != 0
        assert t3.size != 0 and t3_err.size != 0
        assert vis.size != 0 and vis_err.size != 0
        assert vis2.size != 0 and vis2_err.size != 0

        assert flux.shape == (wl_len, 1) and flux_err.shape == (wl_len, 1)
        assert t3.shape == (wl_len, 4) and t3_err.shape == (wl_len, 4)
        assert vis.shape == (wl_len, 6) and vis_err.shape == (wl_len, 6)
        assert vis2.shape == (wl_len, 6) and vis2_err.shape == (wl_len, 6)


# TODO: Add tests for setting the wavelengths and the binning as well
@pytest.mark.parametrize(
    "wavelength", [[3.5] * u.um, [8] * u.um, [3.5, 8] * u.um, [3.5, 8, 10] * u.um]
)
def test_set_data(fits_files: List[Path], wavelength: u.um) -> None:
    """Tests the automatic data procurrment from one
    or multiple (.fits)-files."""
    set_data(fits_files, wavelengths=wavelength, fit_data=["flux", "vis", "vis2", "t3"])
    nwl, nfiles = wavelength.size, len(fits_files)

    flux = OPTIONS.data.flux
    vis = OPTIONS.data.vis if "vis" in OPTIONS.fit.data else OPTIONS.data.vis2
    t3 = OPTIONS.data.t3

    assert flux.value.size != 0 and flux.err.size != 0
    assert flux.value.shape == (nwl, nfiles)
    assert flux.err.shape == (nwl, nfiles)

    assert vis.value.size != 0 and vis.err.size != 0
    assert vis.value.shape == (nwl, 6 * nfiles)
    assert vis.err.shape == (nwl, 6 * nfiles)

    assert t3.value.size != 0 and t3.err.size != 0
    assert t3.value.shape == (nwl, 4 * nfiles)
    assert t3.err.shape == (nwl, 4 * nfiles)

    set_data(fit_data=["flux", "vis", "vis2", "t3"])

    assert flux.value.size == 0 and flux.err.size == 0
    assert vis.value.size == 0 and vis.err.size == 0
    assert t3.value.size == 0 and t3.err.size == 0
