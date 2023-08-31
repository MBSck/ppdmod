import astropy.units as u
import numpy as np

cimport numpy as cnp

cnp.import_array()
DTYPE = np.float64
ctypedef cnp.float64_t DTYPE_t


def calculate_temperature_profile(
        cnp.ndarray[DTYPE_t, ndim=2] radius,
        DTYPE_t eff_radius, DTYPE_t eff_temp, DTYPE_t inner_temp,
        DTYPE_t inner_radius, DTYPE_t q, bint const_temperature):
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
    if const_temperature:
        return np.sqrt(eff_radius/(2*radius))*eff_temp
    return inner_temp*(radius/inner_radius)**(-q)


def calculate_azimuthal_modulation(
        cnp.ndarray[DTYPE_t, ndim=2] xx,
        cnp.ndarray[DTYPE_t, ndim=2] yy, DTYPE_t a, DTYPE_t phi):
    r"""Calculates the azimuthal modulation.

    Parameters
    ----------
    a : astropy.units.one
        The amplitude of the modulation.
    phi : astropy.units.rad
        The phase of the modulation.
    xx : astropy.units.mas
        The x-coordinate grid.
    yy : astropy.units.mas
        The y-coordinate grid.

    Returns
    -------
    azimuthal_modulation : astropy.units.one

    Notes
    -----
    Derived via trigonometry from Lazareff et al. 2017's:

    $ F(r) = F_{0}(r)\cdot\left(1+\sum_{j=1}^{m}()c_{j}\cos(j\phi)+s_{j}\sin(j\phi)\right)$
    """
    return a*np.cos(np.arctan2(yy, xx)-phi)


def calculate_surface_density_profile(
        cnp.ndarray[DTYPE_t, ndim=2] radius,
        cnp.ndarray[DTYPE_t, ndim=2] xx,
        cnp.ndarray[DTYPE_t, ndim=2] yy,
        DTYPE_t a, DTYPE_t phi, DTYPE_t inner_radius,
        DTYPE_t inner_sigma, DTYPE_t p, bint asymmetric):
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
    sigma_profile = inner_sigma*(radius/inner_radius)**(-p)
    if asymmetric:
        return sigma_profile*(1+calculate_azimuthal_modulation(xx, yy, a, phi))
    return sigma_profile
