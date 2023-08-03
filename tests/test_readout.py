from pathlib import Path

import pytest

from ppdmod.readout import ReadoutFits


@pytest.fixture
def readout() -> ReadoutFits:
    """The data of the input files."""
    file = "hd_142666_2022-04-23T03_05_25:2022-04-23T02_28_06_AQUARIUS_FINAL_TARGET_INT.fits"
    return ReadoutFits(Path("data/fits") / file)


def test_readout(readout: ReadoutFits) -> None:
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


def test_wavelength_retrieval(readout: ReadoutFits) -> None:
    """Tests the wavelength retrieval of the (.fits)-files."""
    data4wavelength = readout.get_data_for_wavelength(
        readout.wavelength[50:52], "t3phi")
    assert isinstance(data4wavelength, dict)
    assert len(data4wavelength) == 2
