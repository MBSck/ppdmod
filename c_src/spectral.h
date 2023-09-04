#ifndef SPECTRAL_H
#define SPECTRAL_H


struct Grid {
  double *xx;
  double *yy;
};


double *linspace(float start, float stop, long long dim, double factor);

double *meshgrid(double *linear_grid, long long dim, int axis);

struct Grid grid(
    long long dim, float pixel_size, float pa, float elong, int elliptic);

double *radius(double *xx, double *yy, long long dim);

double *constant_temperature(
    double *radius, float stellar_radius, float stellar_temperature, long long dim);

double *temperature_power_law(
    double *radius, float inner_temp, float inner_radius, float q, long long dim);

double *surface_density_profile(
    double *radius, float inner_radius,
    float inner_sigma, float p, long long dim);

double *azimuthal_modulation(
    double *xx, double *yy, double a, double phi, long long dim);

double *optical_thickness(
    double *surface_density_profile, float opacity, long long dim);

double bb(double temperature, double wavelength);

double *intensity(
    double *temperature_profile, double wavelength, double pixel_size, long long dim);

#endif
