import time
from pathlib import Path
from typing import Optional

import astropy.units as u
import numpy as np
import pandas as pd
import pytest
from astropy.modeling import models

from ppdmod import spectral
from ppdmod import utils
from ppdmod.custom_components import NumericalComponent


RESOLUTION_FILE = Path("resolution.xlsx")
CONST_TEMPERATURE = "Constant Temperature"
TEMPERATURE_POWER = "Temperature Power Law"
AZIMUTHAL_MODULATION = "Azimuthal Modulation"
INTENSITY = "Intensity"

utils.make_workbook(
    RESOLUTION_FILE,
    {
        CONST_TEMPERATURE: ["Dimension [px]",
                            "Python Time [s]",
                            "Cython Time [s]",],
        TEMPERATURE_POWER: ["Dimension [px]",
                            "Python Time [s]",
                            "Cython Time [s]",],
        AZIMUTHAL_MODULATION: ["Dimension [px]",
                               "Python Time [s]",
                               "Cython Time [s]",],
        INTENSITY: ["Dimension [px]",
                    "Python Time [s]",
                    "Cython Time [s]",],
    })

DIMENSION = [2**power for power in range(7, 14)]


@pytest.fixture
def stellar_radius() -> u.Rsun:
    return 1.8*u.Rsun


@pytest.fixture
def stellar_temperature() -> u.K:
    return 7800*u.K


@pytest.fixture
def distance_star() -> u.pc:
    return 150*u.pc


@pytest.fixture
def inner_temperature() -> u.K:
    return 1500*u.K


@pytest.fixture
def inner_radius() -> u.mas:
    return 0.5*u.mas


@pytest.fixture
def q() -> float:
    return 0.5


def calculate_temperature_profile(
        radius: u.mas, distance: u.pc,
        stellar_radius: u.Rsun,
        stellar_temperature: u.K,
        inner_temperature: u.K,
        inner_radius: u.mas,
        q: float,
        const_temperature: bool) -> u.K:
    if const_temperature:
        radius = utils.angular_to_distance(radius, distance)
        return np.sqrt(stellar_radius.to(u.m)/(2*radius))\
            * stellar_temperature
    return inner_temperature*(radius/inner_radius)**(-q)


def calculate_azimuthal_modulation(
        xx: u.mas, yy: u.mas, a: u.one, phi: u.deg) -> None:
    """Calculation of the azimuthal modulation."""
    return a*np.cos(np.arctan2(yy, xx)-phi.to(u.rad))


def calculate_intensity(temp_profile: u.K,
                        wavelength: u.um,
                        pixel_size: Optional[u.Quantity[u.rad]] = None) -> np.ndarray:
    """Calculate the intensity."""
    plancks_law = models.BlackBody(temperature=temp_profile)
    spectral_radiance = plancks_law(wavelength.to(u.m)).to(
        u.erg/(u.cm**2*u.Hz*u.s*u.rad**2))
    return (spectral_radiance*(pixel_size.to(u.rad))**2).to(u.Jy)


@pytest.mark.parametrize("dim", DIMENSION)
def test_calculate_const_temperature(
        dim: int, stellar_radius: u.Rsun,
        stellar_temperature: u.K, distance_star: u.pc,
        inner_temperature: u.K, inner_radius: u.mas, q: float) -> None:
    """Test the calculation of the temperature profile."""
    numerical_component = NumericalComponent(dim=dim, pixel_size=0.1)
    xx, yy = numerical_component._calculate_internal_grid()
    radius = np.hypot(xx, yy)
    stellar_radius_angular =\
        utils.distance_to_angular(stellar_radius.to(u.m), distance_star)

    cython_st = time.time()
    cython_temp = spectral.calculate_const_temperature(
        radius.value,
        stellar_radius_angular.value,
        stellar_temperature.value)
    cython_et = time.time()-cython_st

    python_st = time.time()
    python_temp = calculate_temperature_profile(
        radius, distance_star, stellar_radius,
        stellar_temperature, inner_temperature, inner_radius, q, True)
    python_et = time.time()-python_st

    data = {"Dimension [px]": [dim],
            "Python Time [s]": [python_et],
            "Cython Time [s]": [cython_et]}

    if RESOLUTION_FILE.exists():
        df = pd.read_excel(RESOLUTION_FILE, sheet_name=CONST_TEMPERATURE)
        new_df = pd.DataFrame(data)
        df = pd.concat([df, new_df])
    else:
        df = pd.DataFrame(data)
    with pd.ExcelWriter(RESOLUTION_FILE, engine="openpyxl",
                        mode="a", if_sheet_exists="replace") as writer:
        df.to_excel(writer, sheet_name=CONST_TEMPERATURE, index=False)

    assert np.allclose(cython_temp, python_temp.value)


@pytest.mark.parametrize("dim", DIMENSION)
def test_calculate_temperature_power_law(
        dim: int, stellar_radius: u.Rsun,
        stellar_temperature: u.K, distance_star: u.pc,
        inner_temperature: u.K, inner_radius: u.mas, q: float) -> None:
    """Test the calculation of the temperature profile."""
    numerical_component = NumericalComponent(dim=dim, pixel_size=0.1)
    xx, yy = numerical_component._calculate_internal_grid()
    radius = np.hypot(xx, yy)

    cython_st = time.time()
    cython_temp = spectral.calculate_temperature_power_law(
        radius.value, inner_temperature.value, inner_radius.value, q)
    cython_et = time.time()-cython_st

    python_st = time.time()
    python_temp = calculate_temperature_profile(
        radius, distance_star, stellar_radius,
        stellar_temperature, inner_temperature, inner_radius, q, False)
    python_et = time.time()-python_st

    data = {"Dimension [px]": [dim],
            "Python Time [s]": [python_et],
            "Cython Time [s]": [cython_et]}

    if RESOLUTION_FILE.exists():
        df = pd.read_excel(RESOLUTION_FILE, sheet_name=TEMPERATURE_POWER)
        new_df = pd.DataFrame(data)
        df = pd.concat([df, new_df])
    else:
        df = pd.DataFrame(data)
    with pd.ExcelWriter(RESOLUTION_FILE, engine="openpyxl",
                        mode="a", if_sheet_exists="replace") as writer:
        df.to_excel(writer, sheet_name=TEMPERATURE_POWER, index=False)

    assert np.array_equal(cython_temp, python_temp.value)


@pytest.mark.parametrize("dim", DIMENSION)
def test_calculate_azimuthal_modulation(dim: int) -> None:
    """Test the azimuthal modulation."""
    numerical_component = NumericalComponent(dim=dim, pixel_size=0.1)
    xx, yy = numerical_component._calculate_internal_grid()
    a, phi = 0.5*u.one, 35*u.deg

    cython_st = time.time()
    cython_mod = spectral.calculate_azimuthal_modulation(
        xx.value, yy.value, a.value, phi.to(u.rad).value)
    cython_et = time.time()-cython_st

    python_st = time.time()
    python_mod = calculate_azimuthal_modulation(xx, yy, a, phi)
    python_et = time.time()-python_st

    data = {"Dimension [px]": [dim],
            "Python Time [s]": [python_et],
            "Cython Time [s]": [cython_et]}

    if RESOLUTION_FILE.exists():
        df = pd.read_excel(RESOLUTION_FILE, sheet_name=AZIMUTHAL_MODULATION)
        new_df = pd.DataFrame(data)
        df = pd.concat([df, new_df])
    else:
        df = pd.DataFrame(data)
    with pd.ExcelWriter(RESOLUTION_FILE, engine="openpyxl",
                        mode="a", if_sheet_exists="replace") as writer:
        df.to_excel(writer, sheet_name=AZIMUTHAL_MODULATION, index=False)

    assert np.array_equal(cython_mod, python_mod.value)


@pytest.mark.parametrize("dim", DIMENSION)
def test_calculate_intensity(
        dim: int, stellar_radius: u.Rsun,
        stellar_temperature: u.K, distance_star: u.pc,
        inner_temperature: u.K, inner_radius: u.mas, q: float) -> None:
    """Test the calculation of the intensity."""
    numerical_component = NumericalComponent(dim=dim, pixel_size=0.1)
    xx, yy = numerical_component._calculate_internal_grid()
    radius = np.hypot(xx, yy)
    stellar_radius_angular =\
        utils.distance_to_angular(stellar_radius.to(u.m), distance_star)
    cython_temp = spectral.calculate_const_temperature(
        radius.value,
        stellar_radius_angular.value,
        stellar_temperature.value)
    python_temp = calculate_temperature_profile(
        radius, distance_star, stellar_radius,
        stellar_temperature, inner_temperature, inner_radius, q, True)
    wavelength, pixel_size = 10*u.um, 0.1*u.mas

    cython_st = time.time()
    cython_intensity = spectral.calculate_intensity(
        cython_temp, wavelength.to(u.cm).value,
        pixel_size.to(u.rad).value)
    cython_et = time.time()-cython_st

    python_st = time.time()
    python_intensity = calculate_intensity(python_temp, wavelength, pixel_size)
    python_et = time.time()-python_st

    data = {"Dimension [px]": [dim],
            "Python Time [s]": [python_et],
            "Cython Time [s]": [cython_et]}

    if RESOLUTION_FILE.exists():
        df = pd.read_excel(RESOLUTION_FILE, sheet_name=INTENSITY)
        new_df = pd.DataFrame(data)
        df = pd.concat([df, new_df])
    else:
        df = pd.DataFrame(data)
    with pd.ExcelWriter(RESOLUTION_FILE, engine="openpyxl",
                        mode="a", if_sheet_exists="replace") as writer:
        df.to_excel(writer, sheet_name=INTENSITY, index=False)

    assert np.allclose(cython_intensity, python_intensity.value)
