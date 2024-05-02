from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
import oimodeler as oim

from ppdmod.basic_components import Ring, UniformDisk, Gaussian
from ppdmod.options import OPTIONS
from ppdmod.data import set_data
from ppdmod.utils import compute_vis, compute_t3


@pytest.fixture
def fits_file() -> Path:
    """A MATISSE (.fits)-file."""
    fits_file = "hd_142666_2022-04-23T03_05_25:2022-04-23T02_28_06_HAWAII-2RG_FINAL_TARGET_INT_flux_avg.fits"
    return Path("data/fits/hd142666") / fits_file


@pytest.fixture
def uniform_disk() -> UniformDisk:
    """Initializes a gaussian component."""
    return UniformDisk(dim=512, diam=5)


@pytest.fixture
def gaussian() -> Gaussian:
    """Initializes a gaussian component."""
    return Gaussian(dim=512, fwhm=0.5)


# NOTE:From oimodeler
def corrFlux2Vis2(vcompl):
    nB = vcompl.shape[0]
    norm = np.outer(np.ones(nB-1), vcompl[0, :])
    return np.abs(vcompl[1:, :]/norm)**2


# NOTE:From oimodeler
def corrFlux2T3Phi(vcompl):
    nB = vcompl.shape[0]
    nCP = (nB-1)//3
    norm = np.outer(np.ones(nCP), vcompl[0, :])
    BS = vcompl[1:nCP+1, :]*vcompl[nCP+1:2*nCP+1, :] * \
        np.conjugate(vcompl[2*nCP+1:, :])/norm**3
    return np.rad2deg(np.angle(BS))


# NOTE: Closure phase calculation is identical
def test_ring_vs_oimod(fits_file: Path) -> None:
    """Tests the calculation of uniform disk's visibilities in
    comparison to what oimodeler does."""
    diam, wl = 5, [3.5]*u.um
    set_data([fits_file], wavelengths=wl, fit_data=["vis", "t3"])
    t3 = OPTIONS.data.t3

    ring = Ring(x=5, y=-5, rin=diam/2, thin=True, asymmetric=False)
    oimIring = oim.oimIRing(x=5, y=-5, d=diam)
    oimIring.params["x"].free = oimIring.params["y"].free = True
    vis_ring = ring.compute_complex_vis(t3.u123coord, t3.v123coord, wl)
    vis_oim = oimIring.getComplexCoherentFlux(
            t3.u123coord/wl[0].to(u.m).value, t3.v123coord/wl[0].to(u.m).value, wl[0].to(u.m).value)

    vis_oim = np.append(np.array([1]), vis_oim)
    vis2_oim = corrFlux2Vis2(vis_oim[:, np.newaxis]).reshape(1, -1)
    vis2_ring = compute_vis(vis_ring)**2

    t3_oim = corrFlux2T3Phi(vis_oim[:, np.newaxis])
    t3_ring = compute_t3(vis_ring)

    # assert np.allclose(vis_ring, vis_oim[1:])
    # assert np.allclose(vis2_ring, vis2_oim)

    set_data(fit_data=["vis", "t3"])
