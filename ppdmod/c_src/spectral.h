#ifndef SPECTRAL_H
#define SPECTRAL_H


struct Grid;

double *linspace(float start, float end, int dim, double factor);

double *meshgrid(double *linear_grid, int dim, int axis);

struct Grid grid(
    int dim, float pixel_size, float pa, float elong, int elliptic);

double *radius(double *xx, double *yy, int dim);

double *const_temperature(
    double *radius, float stellar_radius, float stellar_temperature, int dim);

double *temperature_power_law(
    double *radius, float inner_temp, float inner_radius, float q, int dim);

double *surface_density_profile(
    double *radius, float inner_radius,
    float inner_sigma, float p, int dim);

double *azimuthal_modulation(
    double *xx, double *yy, double a, double phi, int dim);

double *optical_thickness(
    double *surface_density_profile, float opacity, int dim);

double bb(double temperature, double wavelength);

double *intensity(
    double *temperature_profile, double wavelength, double pixel_size, int dim);

#endif
