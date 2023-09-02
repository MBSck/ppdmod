#include <iostream>
#include <array>
#include <cmath>
#include <tuple>
#include "spectral.h"


namespace constants {
  const double c = 2.99792458e+10;  // cm/s
  const double c2 = 8.98755179e+20; // cm²/s²
  const double h = 6.62607015e-27;  // erg s
  const double kb = 1.380649e-16;   // erg/K
  const double bb_to_jy = 1.0e+23;  // Jy
}


double *set_linspace(
    float start, float end, int dim, float factor = 1.0) {
  int grid_length = dim;
  double step = (end - start)/dim;
  double *grid = static_cast<double*>(malloc(dim*sizeof(double)));
  for ( int i = 0; i < dim; ++i )
    grid[i] = (start + i * step) * factor;
  return grid;
}


double *create_meshgrid(double *grid, int dim, int axis = 0) {
  double *mesh = static_cast<double*>(malloc(dim*dim*sizeof(double)));
  double temp = 0.0;
  for ( int i = 0; i < dim; ++i ) {
      for ( int j = 0; j < dim; ++j ) {
        if (axis == 0) {
          temp = grid[i];
        } else {
          temp = grid[j];
        }
        mesh[i*dim+j] = temp;
      }
    }
  return mesh;
}

std::tuple<double*, double*> calculate_grid(
    int dim, float pixel_size, float pa, float elong, bool elliptic = false) {
  double *linspace = set_linspace(-0.5, 0.5, dim, 1.0f);
  double *mesh_x = create_meshgrid(linspace, dim, 1);
  double *mesh_y = create_meshgrid(linspace, dim, 0);
  if (elliptic) {
    for ( int i = 0; i < dim*dim; ++i) {
      mesh_x[i] = mesh_x[i]*cos(pa)-mesh_y[i]*sin(pa);
      mesh_y[i] = (mesh_x[i]*sin(pa)+mesh_y[i]*cos(pa))/elong;
    }
  }
  return std::make_tuple(mesh_x, mesh_y);
}


double *calculate_radius(double *xx, double *yy, int dim) {
  double *radius = static_cast<double*>(malloc(dim*dim*sizeof(double)));
  for ( int i = 0; i < dim*dim; ++i ) {
    radius[i] = sqrt(pow(xx[i], 2) + pow(yy[i], 2));
  }
  return radius;
}


double *calculate_const_temperature(
    double *radius, float stellar_radius, float stellar_temperature, int dim) {
  double *const_temperature = static_cast<double*>(malloc(dim*dim*sizeof(double)));
  for ( int i = 0; i < dim*dim; ++i ) {
    const_temperature[i] = stellar_temperature*sqrt(stellar_radius/(2.0*radius[i]));
  }
  return const_temperature;
}

double *calculate_temperature_power_law(
    double *radius, float inner_temp, float inner_radius, float q, int dim) {
  double *temperature_power_law = static_cast<double*>(malloc(dim*dim*sizeof(double)));
  for ( int i = 0; i < dim*dim; ++i ) {
    temperature_power_law[i] = inner_temp*pow(radius[i]/inner_radius, -q);
  }
  return temperature_power_law;
}

double *calculate_surface_density_profile(
    double *radius, float inner_radius,
    float inner_sigma, float p, int dim) {
  double *sigma_profile = static_cast<double*>(malloc(dim*dim*sizeof(double)));
  for ( int i = 0; i < dim*dim; ++i ) {
    sigma_profile[i]= inner_sigma*pow(radius[i]/inner_radius, -p);
  }
  return sigma_profile;
}


double *calculate_azimuthal_modulation(
    double *xx, double *yy, double a, double phi, int dim) {
  double *modulation = static_cast<double*>(malloc(dim*dim*sizeof(double)));
  for ( int i = 0; i < dim*dim; ++i ) {
    modulation[i] = a*cos(atan2(yy[i], xx[i])-phi);
  }
  return modulation;
}


double *calculate_optical_thickness(
    double *surface_density_profile, float opacity, int dim) {
  double *optical_thickness = static_cast<double*>(malloc(dim*dim*sizeof(double)));
  for ( int i = 0; i < dim*dim; ++i ) {
    optical_thickness[i] = 1.0-exp(-surface_density_profile[i]*opacity);
  }
  return optical_thickness;
}


double *calculate_intensity(
    double *temperature_profile, double wavelength, double pixel_size, int dim) {
  double nu = constants::c/wavelength;   // Hz
  double *intensity = static_cast<double*>(malloc(dim*dim*sizeof(double)));
  double temp_val = 0.0;
  for ( int i = 0; i < dim*dim; ++i ) {
      temp_val = 1.0/(exp(constants::h*nu/(constants::kb*temperature_profile[i]))-1.0);
      temp_val = 2.0*constants::h*pow(nu, 3)/constants::c2*temp_val;
      intensity[i] = temp_val*pow(pixel_size, 2)*constants::bb_to_jy;
  }
  return intensity;
}



int main() {
  int dim = 16;
  float pixel_size = 0.1;
  float factor = dim*pixel_size;
  double *xx, *yy;
  std::tie(xx, yy) = calculate_grid(dim, pixel_size, 0.5, 0.33, true);
  double *radius = calculate_radius(xx, yy, dim);
  // for ( int i = 0; i < dim*dim; ++i ) {
  //   std::cout << radius[i] << std::endl;
  // }
  std::cout << radius[137] << std::endl;
  return 0;
}
