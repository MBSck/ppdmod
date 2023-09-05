#ifndef SPECTRAL_H
#define SPECTRAL_H


struct Grid {
  double *xx;
  double *yy;
};


double *linspace(double start, double stop, long long dim, double factor);

double *meshgrid(double *linear_grid, long long dim, int axis);

struct Grid grid(
    long long dim, double pixel_size, double pa, double elong, int elliptic);

double *radius(double *xx, double *yy, long long dim);

double *constant_temperature(
    double *radius, double stellar_radius, double stellar_temperature, long long dim);

double *temperature_power_law(
    double *radius, double inner_temp, double inner_radius, double q, long long dim);

double *surface_density_profile(
    double *radius, double inner_radius,
    double inner_sigma, double p, long long dim);

double *azimuthal_modulation(
    double *xx, double *yy, double a, double phi, long long dim);

double *optical_thickness(
    double *surface_density_profile, double opacity, long long dim);

double bb(double temperature, double wavelength);

double *intensity(
    double *temperature_profile, double wavelength, double pixel_size, long long dim);

double *flat_disk(
    double *radius, double *xx, double *yy, double wavelength, double pixel_size,
    double stellar_radius, double stellar_temperature,
    double inner_temp, double inner_radius, double q, double opacity,
    double inner_sigma, double p, double a, double phi, long long dim,
    int modulated, int const_temperature);

#endif
