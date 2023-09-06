#ifndef SPECTRAL_H
#define SPECTRAL_H

#include <vector>

std::vector<double> constant_temperature(
  const std::vector<double> &radius, double stellar_radius,
  float stellar_temperature, long long dim);

std::vector<double> temperature_power_law(
  const std::vector<double> &radius, float inner_temp,
  float inner_radius, float q, long long dim);

std::vector<double> surface_density_profile(
  const std::vector<double> &radius, float inner_radius,
  float inner_sigma, float p, long long dim);

std::vector<double> azimuthal_modulation(
  const std::vector<double> &xx, const std::vector<double> &yy,
  float a, float phi, long long dim);

std::vector<double> optical_thickness(
  const std::vector<double> &surface_density_profile, double opacity, long long dim);

double bb(double temperature, double wavelength);

std::vector<double> intensity(
  const std::vector<double> &temperature_profile,
  double wavelength, double pixel_size, long long dim);

std::vector<double> flat_disk(
  const std::vector<double> &radius, const std::vector<double> &xx,
  const std::vector<double> &yy, double wavelength, double pixel_size,
  double stellar_radius, float stellar_temperature,
  float inner_temp, float inner_radius, float q, double opacity,
  float inner_sigma, float p, float a, double phi, long long dim,
  int modulated, int const_temperature);

#endif
