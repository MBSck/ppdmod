#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include "spectral.h"


const double c = 2.99792458e+10;  // cm/s
const double c2 = 8.98755179e+20; // cm²/s²
const double h = 6.62607015e-27;  // erg s
const double kb = 1.380649e-16;   // erg/K
const double bb_to_jy = 1.0e+23;  // Jy

struct Grid {
  double *xx;
  double *yy;
};


double *set_linspace(
    float start, float end, int dim, double factor) {
  int grid_length = dim;
  double step = (end - start)/dim;
  double *grid = malloc(sizeof(double)*dim);
  for ( int i = 0; i < dim; ++i )
    grid[i] = (start + i * step) * factor;
  return grid;
}


double *create_meshgrid(double *grid, int dim, int axis) {
  double *mesh = malloc(dim*dim*sizeof(double));
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

struct Grid calculate_grid(
    int dim, float pixel_size, float pa, float elong, int elliptic) {
  struct Grid grid;
  double *linspace = set_linspace(-0.5, 0.5, dim, dim*pixel_size);
  grid.xx = create_meshgrid(linspace, dim, 1);
  grid.xx = create_meshgrid(linspace, dim, 0);

  free(linspace);

  if (elliptic) {
    for ( int i = 0; i < dim*dim; ++i) {
      grid.xx[i] = grid.xx[i]*cos(pa)-grid.yy[i]*sin(pa);
      grid.yy[i] = (grid.xx[i]*sin(pa)+grid.yy[i]*cos(pa))/elong;
    }
  }
  return grid;
}


double *calculate_radius(double *xx, double *yy, int dim) {
  double *radius = malloc(dim*dim*sizeof(double));
  for ( int i = 0; i < dim*dim; ++i ) {
    radius[i] = sqrt(pow(xx[i], 2) + pow(yy[i], 2));
  }
  return radius;
}


double *calculate_const_temperature(
    double *radius, float stellar_radius, float stellar_temperature, int dim) {
  double *const_temperature = malloc(dim*dim*sizeof(double));
  for ( int i = 0; i < dim*dim; ++i ) {
    const_temperature[i] = stellar_temperature*sqrt(stellar_radius/(2.0*radius[i]));
  }
  return const_temperature;
}

double *calculate_temperature_power_law(
    double *radius, float inner_temp, float inner_radius, float q, int dim) {
  double *temperature_power_law = malloc(dim*dim*sizeof(double));
  for ( int i = 0; i < dim*dim; ++i ) {
    temperature_power_law[i] = inner_temp*pow(radius[i]/inner_radius, -q);
  }
  return temperature_power_law;
}

double *calculate_surface_density_profile(
    double *radius, float inner_radius,
    float inner_sigma, float p, int dim) {
  double *sigma_profile = malloc(dim*dim*sizeof(double));
  for ( int i = 0; i < dim*dim; ++i ) {
    sigma_profile[i]= inner_sigma*pow(radius[i]/inner_radius, -p);
  }
  return sigma_profile;
}


double *calculate_azimuthal_modulation(
    double *xx, double *yy, double a, double phi, int dim) {
  double *modulation = malloc(dim*dim*sizeof(double));
  for ( int i = 0; i < dim*dim; ++i ) {
    modulation[i] = a*cos(atan2(yy[i], xx[i])-phi);
  }
  return modulation;
}


double *calculate_optical_thickness(
    double *surface_density_profile, float opacity, int dim) {
  double *optical_thickness = malloc(dim*dim*sizeof(double));
  for ( int i = 0; i < dim*dim; ++i ) {
    optical_thickness[i] = 1.0-exp(-surface_density_profile[i]*opacity);
  }
  return optical_thickness;
}


double *calculate_intensity(
    double *temperature_profile, double wavelength, double pixel_size, int dim) {
  double nu = c/wavelength;   // Hz
  double *intensity = malloc(dim*dim*sizeof(double));
  double temp_val = 0.0;
  for ( int i = 0; i < dim*dim; ++i ) {
      temp_val = 1.0/(exp(h*nu/(kb*temperature_profile[i]))-1.0);
      temp_val = 2.0*h*pow(nu, 3)/c2*temp_val;
      intensity[i] = temp_val*pow(pixel_size, 2)*bb_to_jy;
  }
  return intensity;
}


int main() {
  int dim = 16;
  float pixel_size = 0.1;
  float factor = dim*pixel_size;
  double *xx, *yy;
  struct Grid grid = calculate_grid(dim, pixel_size, 0.5, 0.33, 1);
  double *radius = calculate_radius(grid.xx, grid.yy, dim);
  double *temperature_power_law = calculate_temperature_power_law(radius, 1500.0, 0.5, 0.5, dim);
  return 0;
}
