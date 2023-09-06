#include <iostream>
#include <cmath>
#include <vector>
#include "spectral.h"

namespace constants {
  const double c = 2.99792458e+10;  // cm/s
  const double c2 = 8.98755179e+20; // cm²/s²
  const double h = 6.62607015e-27;  // erg s
  const double kb = 1.380649e-16;   // erg/K
};

namespace conversions {
  const double bb_to_jy = 1.0e+23;  // From cgs units to Jy
}

std::vector<double> constant_temperature(
  const std::vector<double> &radius, double stellar_radius,
  float stellar_temperature, long long dim) {
  std::vector<double> const_temperature;

  std::transform(
    radius.begin(),
    radius.end(),
    std::back_inserter(const_temperature),
    [=](double x) -> double { return stellar_temperature*sqrt(stellar_radius/(2.0*x)); }
  );
  return const_temperature;
}

std::vector<double> temperature_power_law(
  const std::vector<double> &radius, float inner_radius,
  float inner_temp, float q, long long dim) {
  std::vector<double> temperature;

  std::transform(
    radius.begin(),
    radius.end(),
    std::back_inserter(temperature),
    [=](double x) -> double { return inner_temp*pow(x/inner_radius, -q); }
  );

  return temperature;
}

std::vector<double> surface_density_profile(
  const std::vector<double> &radius, float inner_radius,
  double inner_sigma, float p, long long dim) {
  std::vector<double> sigma_profile;

  std::transform(
    radius.begin(),
    radius.end(),
    std::back_inserter(sigma_profile),
    [=](double x) -> double { return inner_sigma*pow(x/inner_radius, -p); }
  );

  return sigma_profile;
}


std::vector<double> azimuthal_modulation(
  const std::vector<double> &xx, const std::vector<double> &yy,
  float a, double phi, long long dim) {
  std::vector<double> modulation;

  for ( long long i = 0; i < dim*dim; ++i ) {
    modulation.push_back(a*cos(atan2(yy[i], xx[i])-phi));
  }
  return modulation;
}


std::vector<double> optical_thickness(
  const std::vector<double> &surface_density, double opacity, long long dim) {
  std::vector<double> thickness;
  std::transform(
    surface_density.begin(),
    surface_density.end(),
    std::back_inserter(thickness),
    [=](double x) -> double { return 1.0-exp(-x*opacity); }
  );
  return thickness;
}

double bb(double temperature, double wavelength) {
  double nu = constants::c/wavelength;   // Hz
  return (2.0*constants::h*pow(nu, 3)/constants::c2)*(1.0/(exp(constants::h*nu/(constants::kb*temperature))-1.0));
}


std::vector<double> intensity(
  const std::vector<double> &temperature_profile, double wavelength, double pixel_size, long long dim) {
  std::vector<double> intensity;

  std::transform(
    temperature_profile.begin(),
    temperature_profile.end(),
    std::back_inserter(intensity),
    [=](double x) -> double {
      return bb(x, wavelength)*pow(pixel_size, 2)*conversions::bb_to_jy ; }
  );
  return intensity;
}

std::vector<double> flat_disk(
  const std::vector<double> &radius, const std::vector<double> &xx,
  const std::vector<double> &yy, double wavelength, double pixel_size,
  double stellar_radius, float stellar_temperature,
  float inner_temp, float inner_radius, float q, double opacity,
  float inner_sigma, float p, float a, double phi, long long dim,
  bool modulated, bool const_temperature) {
  double modulation = 1.0;
  double radius_val, temperature, surface_density, thickness, blackbody;
  std::vector<double> brightness;

  for ( long long i = 0; i < dim*dim; ++i ) {
    radius_val = radius[i];
    if (const_temperature) {
      temperature = stellar_temperature*sqrt(stellar_radius/(2.0*radius_val));
    } else {
      temperature = inner_temp*pow(radius_val/inner_radius, -q);
    }
    surface_density = inner_sigma*pow(radius_val/inner_radius, -p);
    if (modulated) {
      modulation = a*cos(atan2(yy[i], xx[i])-phi);
    }
    thickness = 1.0-exp(-surface_density*modulation*opacity);
    blackbody = bb(temperature, wavelength)*pow(pixel_size, 2)*conversions::bb_to_jy;
    brightness.push_back(blackbody*thickness);
  }
  return brightness;
}

int main() {
  return 0;
}
