#ifndef SPECTRAL_H
#define SPECTRAL_H

#include <tuple>

namespace constants {
  extern const double c;
  extern const double c2;
  extern const double h;
  extern const double kb;
  extern const double bb_to_jy;
}

double *set_linspace(float start, float end, int dim, float factor);

double *create_meshgrid(double *grid, int dim, int axis);

std::tuple<double*, double*> calculate_grid(
    int dim, float pixel_size, float pa, float elong, bool elliptic);

double *calculate_radius(double *xx, double *yy, int dim);

double *calculate_const_temperature(
    double *radius, float stellar_radius, float stellar_temperature, int dim);

double *calculate_temperature_power_law(
    double *radius, float inner_temp, float inner_radius, float q, int dim);

double *calculate_surface_density_profile(
    double *radius, float inner_radius,
    float inner_sigma, float p, int dim);

double *calculate_azimuthal_modulation(
    double *xx, double *yy, double a, double phi, int dim);

double *calculate_optical_thickness(
    double *surface_density_profile, float opacity, int dim);

double *calculate_intensity(
    double *temperature_profile, double wavelength, double pixel_size, int dim);

#endif
