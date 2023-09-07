import numpy as np

cimport numpy as cnp
cimport cython
from libc.math cimport sin, cos, atan2, sqrt, exp, pow

cnp.import_array()
DTYPE = np.float64
ctypedef cnp.float64_t DTYPE_t


cdef double c = 2.99792458e+10   # cm/s
cdef double c2 = 8.98755179e+20  # cm²/s²
cdef double h = 6.62607015e-27   # erg s
cdef double kb = 1.380649e-16    # erg/K
cdef double bb_to_jy = 1.0e+23   # Jy


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
def grid(int dim, float pixel_size,
         float elong, double pa, int elliptic):
    """Calculates the model grid.

    Parameters
    ----------
    dim : float, optional
    pixel_size : float, optional

    Returns
    -------
    xx : astropy.units.mas
        The x-coordinate grid.
    yy : astropy.units.mas
        The y-coordinate grid.
    """
    linspace = np.empty((dim, ), dtype=DTYPE)
    x_arr = np.empty((dim, dim), dtype=DTYPE)
    y_arr = np.empty((dim, dim), dtype=DTYPE)
    radius = np.empty((dim, dim), dtype=DTYPE)

    cdef double[::1] linspace_view = linspace
    cdef double[:, ::1] x_arr_view = x_arr
    cdef double[:, ::1] y_arr_view = y_arr
    cdef double[:, ::1] radius_view = radius

    cdef double step =  1.0/dim
    cdef float factor = dim*pixel_size

    cdef Py_ssize_t x
    for x in range(dim):
        linspace_view[x] = (-0.5 + x * step) * factor;

    cdef Py_ssize_t y
    cdef double linspace_val
    for x in range(dim):
        for y in range(dim):
            linspace_val = linspace_view[y]
            x_arr_view[x, y] = linspace_val
            y_arr_view[y, x] = linspace_val

    cdef double xx_val, yy_val
    cdef double temp_x, temp_y
    for x in range(dim):
        for y in range(dim):
            xx_val, yy_val = x_arr_view[x, y], y_arr_view[x, y]
            if elliptic:
                temp_x = xx_val*cos(pa)-yy_val*sin(pa)
                temp_y = (xx_val*sin(pa)+yy_val*cos(pa))/elong
            else:
                temp_x, temp_y = xx_val, yy_val
            x_arr_view[x, y], y_arr_view[x, y] = temp_x, temp_y
    return x_arr, y_arr


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.cpow(True)
@cython.initializedcheck(False)
def radius(double[:, ::1] xx, double[:, ::1] yy):
    """Calculates the model radius."""
    cdef Py_ssize_t x_max = xx.shape[0]
    cdef Py_ssize_t y_max = yy.shape[1]

    radius = np.empty((x_max, y_max), dtype=DTYPE)
    cdef double[:, ::1] radius_view = radius

    cdef double xx_val, yy_val
    cdef Py_ssize_t x, y
    for x in range(x_max):
        for y in range(y_max):
            xx_val, yy_val = xx[x, y], yy[x, y]
            radius_view[x, y] = sqrt(pow(xx_val, 2)+pow(yy_val, 2))
    return radius


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
def const_temperature(
        double[:, ::1] radius, double stellar_radius, double stellar_temperature):
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

    temperature = np.empty((x_max, y_max), dtype=DTYPE)
    cdef double[:, ::1] temperature_view = temperature

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
@cython.cpow(True)
@cython.initializedcheck(False)
def temperature_power_law(
        double[:, ::1] radius, double inner_temp,
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
    cdef double[:, ::1] temperature_view = temperature

    cdef Py_ssize_t x, y
    cdef double radius_val, temp_val
    for x in range(x_max):
        for y in range(y_max):
            radius_val = radius[x, y]
            temp_val = inner_temp*pow(radius_val/inner_radius, -q)
            temperature_view[x, y] = temp_val
    return temperature


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.cpow(True)
@cython.initializedcheck(False)
def surface_density_profile(
        double[:, ::1] radius, double inner_radius,
        double inner_sigma, double p):
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
    cdef Py_ssize_t x_max = radius.shape[0]
    cdef Py_ssize_t y_max = radius.shape[1]

    sigma_profile = np.empty((x_max, y_max), dtype=DTYPE)
    cdef double[:, ::1] sigma_profile_view = sigma_profile

    cdef Py_ssize_t x, y
    cdef double radius_val, temp_val
    for x in range(x_max):
        for y in range(y_max):
            radius_val = radius[x, y]
            temp_val = inner_sigma*pow(radius_val/inner_radius, -p)
            sigma_profile_view[x, y] = temp_val
    return sigma_profile


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
def azimuthal_modulation(
        double[:, ::1] xx, double[:, ::1] yy, double a, double phi):
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

    modulation = np.empty((x_max, y_max), dtype=DTYPE)
    cdef double[:, ::1] modulation_view = modulation

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
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def optical_thickness(
        double[:, ::1] surface_density_profile, double opacity):
    """Calculates the optical depth from the surface density and
    opacity profiles."""
    cdef Py_ssize_t x_max = surface_density_profile.shape[0]
    cdef Py_ssize_t y_max = surface_density_profile.shape[1]

    optical_thickness = np.empty((x_max, y_max), dtype=DTYPE)
    cdef double[:, ::1] optical_thickness_view = optical_thickness

    cdef Py_ssize_t x, y
    cdef double surface_density_val, temp_val
    for x in range(x_max):
        for y in range(y_max):
            surface_density_val = surface_density_profile[x, y]
            temp_val = 1.0-exp(-surface_density_val*opacity)
            optical_thickness_view[x, y] = temp_val
    return optical_thickness


@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.cpow(True)
@cython.initializedcheck(False)
def intensity(
        double[:, ::1] temperature_profile, double wavelength, double pixel_size):
    """Calculates the blackbody_profile via Planck's law and the
    emissivity_factor for a given wavelength, temperature- and
    dust surface density profile.

    Parameters
    ----------
    wavelengths : astropy.units.cm
        Wavelength value(s).
    temp_profile : astropy.units.K
        Temperature profile.
    pixel_size : astropy.units.rad
        The pixel size.

    Returns
    -------
    intensity : astropy.units.Jy
        Intensity per pixel [Jy/px]
    """
    cdef double nu = c/wavelength    # Hz
    cdef Py_ssize_t x_max = temperature_profile.shape[0]
    cdef Py_ssize_t y_max = temperature_profile.shape[1]

    intensity = np.empty((x_max, y_max), dtype=DTYPE)
    cdef double[:, ::1] intensity_view = intensity

    cdef Py_ssize_t x, y
    cdef double temperature_val, temp_val
    for x in range(x_max):
        for y in range(y_max):
            temperature_val = temperature_profile[x, y]
            temp_val = 1.0/(exp(h*nu/(kb*temperature_val))-1.0)
            temp_val = 2.0*h*pow(nu, 3)/c2*temp_val
            intensity_view[x, y] = temp_val*pow(pixel_size, 2)*bb_to_jy
    return intensity
