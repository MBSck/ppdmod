from typing import List
from pathlib import Path

import astropy.units as u
import emcee
import numpy as np
import pytest
from scipy.optimize import minimize

from ppdmod import data
from ppdmod.custom_components import Star, AsymmetricSDGreyBodyContinuum
from ppdmod.model import Model
from ppdmod.plot import plot_corner, plot_chains, plot_model, plot_observed_vs_model


PLOT_DIR = Path("plots")
if not PLOT_DIR.exists():
    PLOT_DIR.mkdir()


@pytest.fixture
def dim() -> int:
    """The dimension of the model."""
    return 2048


@pytest.fixture
def pixel_size() -> int:
    """The pixel size of the model."""
    return 0.1*u.mas


@pytest.fixture
def position_angle() -> u.deg:
    """The position angle of the model."""
    return 45*u.deg


@pytest.fixture
def axis_ratio() -> u.one:
    """The axis ratio of the model."""
    return 1.6


@pytest.fixture
def wavelength() -> u.m:
    """A wavelenght grid."""
    return (13.000458e-6*u.m).to(u.um)


@pytest.fixture
def wavelengths() -> u.um:
    """A wavelength grid."""
    return ([8.28835527e-06, 1.02322101e-05, 13.000458e-6]*u.m).to(u.um)


@pytest.fixture
def fits_files() -> Path:
    """A MATISSE (.fits)-file."""
    files = [
        "hd_142666_2022-04-23T03_05_25:2022-04-23T02_28_06_AQUARIUS_FINAL_TARGET_INT.fits",
        "hd_142666_2022-04-21T07_18_22:2022-04-21T06_47_05_AQUARIUS_FINAL_TARGET_INT.fits"]
    return [Path("data/fits") / file for file in files]


@pytest.fixture
def model(dim: int, pixel_size: u.mas,
          axis_ratio: u.one, position_angle: u.deg) -> Model:
    """Creates a model."""
    static_params = {"dim": dim, "dist": 145,
                     "eff_temp": 7800, "eff_radius": 1.8}
    star = Star(**static_params)
    inner_ring = AsymmetricSDGreyBodyContinuum(
        **static_params,
        rin=25, rout=30, a=0.3, phi=33,
        pixel_size=pixel_size.value, pa=position_angle,
        elong=axis_ratio, inner_sigma=2000, kappa_abs=1000,
        kappa_cont=3000, cont_weight=0.5, p=0.5)
    inner_ring.optically_thick = True
    outer_ring = AsymmetricSDGreyBodyContinuum(
        **static_params,
        rin=33, a=0.3, phi=33,
        pixel_size=pixel_size.value, pa=position_angle,
        elong=axis_ratio, inner_sigma=2000, kappa_abs=1000,
        kappa_cont=3000, cont_weight=0.5, p=0.5)
    outer_ring.optically_thick = True
    return Model([star, inner_ring, outer_ring])


def log_prior(theta):
    """Log prior. From emcee homepage."""
    m, b, log_f = theta
    if -5.0 < m < 0.5 and 0.0 < b < 10.0 and -10.0 < log_f < 1.0:
        return 0.0
    return -np.inf


def log_likelihood(theta, x, y, yerr):
    """Log likelihood. From emcee homepage."""
    m, b, log_f = theta
    model = m * x + b
    sigma2 = yerr**2 + model**2 * np.exp(2 * log_f)
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))


def log_probability(theta, x, y, yerr):
    """Log probability. From emcee homepage."""
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)


@pytest.fixture
def sampler() -> None:
    """Runs emcee. Example from emcee homepage."""

    # Choose the "true" parameters.
    m_true = -0.9594
    b_true = 4.294
    f_true = 0.534

    # Generate some synthetic data from the model.
    N = 50
    x = np.sort(10 * np.random.rand(N))
    yerr = 0.1 + 0.5 * np.random.rand(N)
    y = m_true * x + b_true
    y += np.abs(f_true * y) * np.random.randn(N)
    y += yerr * np.random.randn(N)

    np.random.seed(42)
    nll = lambda *args: -log_likelihood(*args)
    initial = np.array([m_true, b_true, np.log(f_true)])\
        + 0.1 * np.random.randn(3)
    soln = minimize(nll, initial, args=(x, y, yerr))
    pos = soln.x + 1e-4 * np.random.randn(32, 3)
    nwalkers, ndim = pos.shape

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability, args=(x, y, yerr)
    )
    sampler.run_mcmc(pos, 5000, progress=True)
    return sampler


@pytest.fixture
def labels() -> List[str]:
    """Labels for emcee."""
    return ["m", "b", "log(f)"]


def test_plot_corner(sampler: np.ndarray,
                     labels: List[str]) -> None:
    """Tests the plot corner function."""
    plot_corner(sampler, labels,
                savefig=PLOT_DIR / "corner.pdf")


def test_plot_chains(sampler: np.ndarray,
                     labels: List[str]) -> None:
    """Tests the plot chains function."""
    plot_chains(sampler, labels,
                savefig=PLOT_DIR / "chains.pdf")


def test_plot_model(model: Model, dim: int,
                    pixel_size: u.mas, wavelength: u.um) -> None:
    """Tests the plot model function."""
    plot_model(dim, pixel_size, model, wavelength, savefig=PLOT_DIR / "model.pdf")


def test_plot_observed_vs_model(fits_files: List[Path], model: Model, pixel_size: u.mas,
                                axis_ratio: u.one, position_angle: u.deg,
                                wavelengths: u.um) -> None:
    """Tests the plot observed vs model function."""
    data.set_fit_wavelengths(wavelengths)
    data.set_data(fits_files)
    plot_observed_vs_model(model, pixel_size, axis_ratio, position_angle,
                           savefig=PLOT_DIR / "observed_vs_model.pdf")
    data.set_fit_wavelengths()
    data.set_data()
