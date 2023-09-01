import numpy as np

cimport numpy as cnp
cimport cython
from libc.math cimport cos, atan2, sqrt, pow

cnp.import_array()
DTYPE = np.float64
ctypedef cnp.float64_t DTYPE_t


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
def calculate_const_temperature(
        double [:, :] radius, double stellar_radius, double stellar_temperature):
    """Calculates the temperature profile.

    Can be specified to be either as a r^q power law or an a
    constant/idealised temperature profile derived from the star's
    temperature, its radius and the observer's distance to the star.
    It will then be only contingent on those values.

    Parameters
    ----------
    radius : astroy.units.mas
        The radial grid.
    stellar_radius : astropy.units.mas
        The radius of the star.
    stellar_temperature : astropy.units.K
        The effective temperature of the star.
    

    Returns
    -------
    temperature_profile : astropy.units.K

    Notes
    -----
    In case of a radial power law the formula is

    .. math:: T = T_0 * (1+\\frac{r}{R_0})^\\q.

    In case of a constant grey body profile the stellar radius is
    calculated from its lumionsity via

    .. math:: R_* = \\sqrt{\\frac{L_*}{4\\pi\\sigma_sb\\T_*^4}}.

    And with this the individual grain's temperature profile is

    .. math:: T_{grain} = \\sqrt{\\frac{R_*}{2r}}\\cdot T_*.
    """
    cdef Py_ssize_t x_max = radius.shape[0]
    cdef Py_ssize_t y_max = radius.shape[1]

    temperature = np.zeros((x_max, y_max), dtype=np.float64)
    cdef double [:, :] temperature_view = temperature

    cdef Py_ssize_t x, y
    cdef double radius_val, temp_val
    for x in range(x_max):
        for y in range(y_max):
            radius_val = radius[x, y]
            temp_val = sqrt(stellar_radius/(2.0*radius_val))*stellar_temperature
            temperature_view[x, y] = temp_val
    return temperature


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.cpow(True)
def calculate_temperature_power_law(
        double [:, :] radius, double inner_temp,
        double inner_radius, double q):
    """Calculates the temperature profile.

    Can be specified to be either as a r^q power law or an a
    constant/idealised temperature profile derived from the star's
    temperature, its radius and the observer's distance to the star.
    It will then be only contingent on those values.

    Parameters
    ----------
    radius : astroy.units.m
        The radial grid.
    eff_radius : astropy.units.m
        The radius of the star.
    eff_temp : astropy.units.K
        The effective temperature of the star.
    inner_temp : astropy.units.K
        The temperature of the innermost grain.
    inner_radius : astropy.units.m
        The radius of the innermost grain.
    q : astropy.units.one
    const_temperature : bool
    

    Returns
    -------
    temperature_profile : astropy.units.K

    Notes
    -----
    In case of a radial power law the formula is

    .. math:: T = T_0 * (1+\\frac{r}{R_0})^\\q.

    In case of a constant grey body profile the stellar radius is
    calculated from its lumionsity via

    .. math:: R_* = \\sqrt{\\frac{L_*}{4\\pi\\sigma_sb\\T_*^4}}.

    And with this the individual grain's temperature profile is

    .. math:: T_{grain} = \\sqrt{\\frac{R_*}{2r}}\\cdot T_*.
    """
    cdef Py_ssize_t x_max = radius.shape[0]
    cdef Py_ssize_t y_max = radius.shape[1]

    temperature = np.empty((x_max, y_max), dtype=DTYPE)
    cdef double [:, :] temperature_view = temperature

    cdef Py_ssize_t x, y
    cdef double radius_val, temp_val
    for x in range(x_max):
        for y in range(y_max):
            radius_val = radius[x, y]
            temp_val = (radius_val/inner_radius)**(-q)*inner_temp
            temperature_view[x, y] = temp_val
    return temperature


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
def calculate_azimuthal_modulation(
        double [:, :] xx, double [:, :] yy, double a, double phi):
    r"""Calculates the azimuthal modulation.

    Parameters
    ----------
    xx : astropy.units.mas
        The x-coordinate grid.
    yy : astropy.units.mas
        The y-coordinate grid.
    a : astropy.units.one
        The amplitude of the modulation.
    phi : astropy.units.rad
        The phase of the modulation.

    Returns
    -------
    azimuthal_modulation : astropy.units.one

    Notes
    -----
    Derived via trigonometry from Lazareff et al. 2017's:

    $ F(r) = F_{0}(r)\cdot\left(1+\sum_{j=1}^{m}()c_{j}\cos(j\phi)+s_{j}\sin(j\phi)\right)$
    """
    cdef Py_ssize_t x_max = xx.shape[0]
    cdef Py_ssize_t y_max = xx.shape[1]

    modulation = np.zeros((x_max, y_max), dtype=np.float64)
    cdef double [:, :] modulation_view = modulation

    cdef Py_ssize_t x, y
    cdef double xx_val, yy_val, temp_val
    for x in range(x_max):
        for y in range(y_max):
            xx_val = xx[x, y]
            yy_val = yy[x, y]
            temp_val = a*cos(atan2(yy_val, xx_val)-phi)
            modulation_view[x, y] = temp_val
    return modulation


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.cpow(True)
def calculate_surface_density_profile(
        double [:, :] radius, double [:, :] xx,
        double [:, :] yy, double a, double phi,
        double inner_radius, double inner_sigma, double p):
    """Calculates the surface density profile.

    This can be azimuthally varied if so specified.

    Parameters
    ----------
    radius : astropy.units.mas
        The radial grid.

    Returns
    -------
    surface_density_profile : astropy.units.g/astropy.units.cm**2

    Notes
    -----
    """
    cdef Py_ssize_t x_max = xx.shape[0]
    cdef Py_ssize_t y_max = xx.shape[1]

    sigma_profile = np.zeros((x_max, y_max), dtype=np.float64)
    cdef double [:, :] sigma_profile_view = sigma_profile

    cdef Py_ssize_t x, y
    cdef double radius_val, temp_val
    for x in range(x_max):
        for y in range(y_max):
            radius_val = radius[x, y]
            temp_val = inner_sigma*(radius_val/inner_radius)**(-p)
            sigma_profile_view[x, y] = temp_val
    return sigma_profile
