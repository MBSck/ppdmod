from typing import Optional

import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.modeling import models


@u.quantity_input
def calculate_effective_baselines(uv_coords: u.m,
                                  axis_ratio: u.dimensionless_unscaled,
                                  pos_angle: u.rad,
                                  wavelength: u.um) -> u.dimensionless_unscaled:
    """Calculates the effective baselines from the projected baselines in MegaLambdas

    Parameters
    ----------
    uv_coords: u.m
        The (u, v)-coordinates
    axis_ratio: u.dimensionless_unscaled
        The axis ratio of the ellipse
    pos_angle: u.rad
        The positional angle of the object
    wavelength: u.um
        The wavelength to calulate the MegaLambda
    """
    if uv_coords.shape[0] == 3:
        u_coords, v_coords = map(lambda x: x.squeeze(),
                                 np.split(uv_coords, 2, axis=2))
    else:
        u_coords, v_coords = map(lambda x: x.squeeze(),
                                 np.split(uv_coords, 2, axis=1))

    projected_baselines = np.sqrt(u_coords**2+v_coords**2)
    projected_baselines_angle = np.arctan2(u_coords, v_coords)\
            .to(u.rad, equivalencies=u.dimensionless_angles())
    atd = np.arctan2(np.sin(projected_baselines_angle-pos_angle),
                     (np.cos(projected_baselines_angle-pos_angle)))
    u_coords_eff = projected_baselines*(np.cos(atd)*np.cos(pos_angle)\
                                       -axis_ratio*np.sin(atd)*np.sin(pos_angle))
    v_coords_eff = projected_baselines*(np.cos(atd)*np.sin(pos_angle)\
                                       +axis_ratio*np.sin(atd)*np.cos(pos_angle))
    return np.sqrt(u_coords_eff**2+v_coords_eff**2)/wavelength.value


@u.quantity_input
def _convert_orbital_radius_to_parallax(orbital_radius: u.m,
                                        distance: Optional[u.Quantity[u.pc]] = None) -> u.mas:
    """Calculates the parallax [astropy.units.mas] from the orbital radius
    [astropy.units.m]. The formula for the angular diameter is used

    Parameters
    ----------
    orbital_radius: astropy.units.Quantity
        The orbital radius [astropy.units.m]
    distance: astropy.units.Quantity
        The distance to the star from the observer [astropy.units.pc]

    Returns
    -------
    parallax: astropy.units.Quantity
        The angle of the orbital radius [astropy.units.mas]
    """
    return (1*u.rad).to(u.mas)*(orbital_radius.to(u.m)/distance.to(u.m))


@u.quantity_input
def _convert_parallax_to_orbital_radius(parallax: u.mas,
                                        distance: Optional[u.Quantity[u.pc]] = None) -> u.m:
    """Calculates the orbital radius [astropy.units.m] from the parallax
    [astropy.units.mas]. The formula for the angular diameter is used

    Parameters
    ----------
    parallax: u.mas
        The angle of the orbital radius
    distance: u.pc
        The distance to the star from the observer

    Returns
    -------
    orbital_radius: u.m
        The orbital radius
    """
    return (parallax*distance.to(u.m))/(1*u.rad).to(u.mas)


@u.quantity_input
def _calculate_stellar_radius(luminosity_star: u.W, effective_temperature: u.K) -> u.m:
    """Calculates the stellar radius [astropy.units.m] from its attributes.
    Only for 'delta_component' functionality

    Parameters
    ----------
    luminosity_star: u.W
        The luminosity of the star
    effective_temperature: u.K
        The effective temperature of the star

    Returns
    -------
    stellar_radius: astropy.units.quantity
        the star's radius [astropy.units.m]
    """
    return np.sqrt(luminosity_star/(4*np.pi*const.sigma_sb*effective_temperature**4))


# TODO: Make test with Jozsef's values
@u.quantity_input
def stellar_flux(wavelength: u.um, effective_temperature: u.K,
                 distance: u.pc, luminosity_star: u.W) -> u.Jy:
    """Calculates the stellar flux from the distance and its radius.
    Only for 'delta_component' functionality

    Parameters
    ----------
    wavelength: u.um
        The wavelength to be used for the BlackBody calculation
    luminosity_star: u.W
        The luminosity of the star
    effective_temperature: u.K
        The effective temperature of the star
    distance: astropy.units.Quantity
        The distance to the star from the observer

    Returns
    -------
    stellar_flux: u.Jy
        The star's flux
    """
    plancks_law = models.BlackBody(temperature=effective_temperature)
    spectral_radiance = plancks_law(wavelength).to(u.erg/(u.cm**2*u.Hz*u.s*u.mas**2))
    stellar_radius = _calculate_stellar_radius(luminosity_star, effective_temperature)
    # TODO: Check if that can be used in this context -> The conversion
    stellar_radius_angular = _convert_orbital_radius_to_parallax(stellar_radius, distance)
    return (spectral_radiance*np.pi*(stellar_radius_angular)**2).to(u.Jy)


@u.quantity_input
def _calculate_inner_radius(inner_temperature: u.K,
                            distance: u.pc, luminosity_star: u.W) -> u.mas:
    """Calculates the sublimation radius at the inner rim of the disc

    Returns
    -------
    sublimation_radius: u.K
        The sublimation radius
    distance: u.pc
        The distance to the star from the observer
    luminosity_star: u.W
        The luminosity of the star
    """
    radius = np.sqrt(luminosity_star/(4*np.pi*const.sigma_sb*inner_temperature**4))
    return _convert_orbital_radius_to_parallax(radius, distance)


@u.quantity_input
def _calculate_inner_temperature(inner_radius: u.mas,
                                 distance: u.pc, luminosity_star: u.W) -> u.K:
    """Calculates the sublimation temperature at the inner rim of the disc

    Parameters
    ----------
    inner_radius: astropy.units.Quantity
        The inner radius of the disc [astropy.units.mas]
    luminosity_star: astropy.units.Quantity
        The luminosity of the star [astropy.units.W]

    Returns
    -------
    sublimation_temperature: astropy.units.Quantity
        The sublimation temperature [astropy.units.K]
    """
    if inner_radius.unit == u.mas:
        inner_radius = _convert_parallax_to_orbital_radius(inner_radius, distance)
    return (luminosity_star/(4*np.pi*const.sigma_sb*inner_radius**2))**(1/4)


@u.quantity_input
def temperature_gradient(radius: u.mas, power_law_exponent: u.dimensionless_unscaled,
                         inner_radius: u.mas, inner_temperature: u.K) -> u.K:
    """Calculates the temperature gradient

    Parameters
    ----------
    radius: u.mas
        An array containing all the points for the radius extending outwards
    power_law_exponent: u.dimensionless_unscaled
        A float specifying the power law exponent of the temperature gradient "q"
    inner_radius: u.K
        The inner radius of the object, if not given then the sublimation radius is
        used
    inner_temperature: u.K
        The temperature of the inner rim

    Returns
    -------
    temperature_gradient: u.K
    """
    temperature = models.PowerLaw1D().evaluate(radius, inner_temperature,
                                               inner_radius, power_law_exponent)
    temperature[temperature == np.inf] = 0.*temperature.unit
    return temperature


@u.quantity_input
def optical_depth_gradient(radius: u.mas,
                           power_law_exponent: u.dimensionless_unscaled,
                           inner_radius: u.mas,
                           inner_optical_depth: u.dimensionless_unscaled
                           ) -> u.dimensionless_unscaled:
    """Calculates the optical depth gradient

    Parameters
    ----------
    radius: u.mas
        An array containing all the points for the radius extending outwards
    power_law_exponent: u.dimensionless_unscaled
        A float specifying the power law exponent of the temperature gradient "q"
    inner_radius: u.mas
        The inner radius of the object, if not given then the sublimation radius is
        used
    inner_optical_depth: u.dimensionless_unscaled
        The optical depth at the inner radius

    Returns
    -------
    optical_depth_gradient: u.dimensionless_unscaled
    """
    optical_depth = models.PowerLaw1D().evaluate(radius, inner_optical_depth,
                                                 inner_radius, power_law_exponent)
    optical_depth[optical_depth == np.inf] = 0.*optical_depth.unit
    return optical_depth


@u.quantity_input
def flux_per_pixel(wavelength: u.um, temperature_distribution: u.K,
                   optical_depth: u.dimensionless_unscaled, pixel_size: u.mas) -> u.Jy:
    """Calculates the total flux of the model

    Parameters
    ----------
    wavelength: u.um
        The wavelength to be used for the BlackBody calculation
    temperature: u.K
        The temperature distribution of the disc
    optical_depth: u.dimensionless_unscaled
        The optical depth of the disc
    pixel_size: u.mas/px
        The pixel size determined by the field of view and the number of pixels

    Returns
    -------
    flux: u.Jy/px
        The object's flux per pixel
    """
    plancks_law = models.BlackBody(temperature=temperature_distribution)
    # NOTE: Converts sr to mas**2. Field of view = sr or mas**2
    spectral_radiance = plancks_law(wavelength).to(u.erg/(u.cm**2*u.Hz*u.s*u.mas**2))
    flux_per_pixel = spectral_radiance*pixel_size**2
    return (flux_per_pixel.to(u.Jy))*(1-np.exp(-optical_depth))


if __name__ == "__main__":
    ...
