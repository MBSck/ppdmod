#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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


double *linspace(
    float start, float end, int dim, double factor) {
  int grid_length = dim;
  double step = (end - start)/dim;
  double *linear_grid = malloc(sizeof(double)*dim);
  for ( int i = 0; i < dim; ++i )
    linear_grid[i] = (start + i * step) * factor;
  return linear_grid;
}


double *meshgrid(double *grid, int dim, int axis) {
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

struct Grid grid(
    int dim, float pixel_size, float pa, float elong, int elliptic) {
  struct Grid grid;
  double *x = linspace(-0.5, 0.5, dim, dim*pixel_size);
  grid.xx = meshgrid(x, dim, 1);
  grid.yy = meshgrid(x, dim, 0);

  free(x);

  if (elliptic) {
    for ( int i = 0; i < dim*dim; ++i) {
      grid.xx[i] = grid.xx[i]*cos(pa)-grid.yy[i]*sin(pa);
      grid.yy[i] = (grid.xx[i]*sin(pa)+grid.yy[i]*cos(pa))/elong;
    }
  }
  return grid;
}


double *radius(double *xx, double *yy, int dim) {
  double *radius = malloc(dim*dim*sizeof(double));
  for ( int i = 0; i < dim*dim; ++i ) {
    radius[i] = sqrt(pow(xx[i], 2) + pow(yy[i], 2));
  }
  return radius;
}


double *const_temperature(
    double *radius, float stellar_radius, float stellar_temperature, int dim) {
  double *const_temperature = malloc(dim*dim*sizeof(double));
  for ( int i = 0; i < dim*dim; ++i ) {
    const_temperature[i] = stellar_temperature*sqrt(stellar_radius/(2.0*radius[i]));
  }
  return const_temperature;
}

double *temperature_power_law(
    double *radius, float inner_temp, float inner_radius, float q, int dim) {
  double *temperature_power_law = malloc(dim*dim*sizeof(double));
  for ( int i = 0; i < dim*dim; ++i ) {
    temperature_power_law[i] = inner_temp*pow(radius[i]/inner_radius, -q);
  }
  return temperature_power_law;
}

double *surface_density_profile(
    double *radius, float inner_radius,
    float inner_sigma, float p, int dim) {
  double *sigma_profile = malloc(dim*dim*sizeof(double));
  for ( int i = 0; i < dim*dim; ++i ) {
    sigma_profile[i]= inner_sigma*pow(radius[i]/inner_radius, -p);
  }
  return sigma_profile;
}


double *azimuthal_modulation(
    double *xx, double *yy, double a, double phi, int dim) {
  double *modulation = malloc(dim*dim*sizeof(double));
  for ( int i = 0; i < dim*dim; ++i ) {
    modulation[i] = a*cos(atan2(yy[i], xx[i])-phi);
  }
  return modulation;
}


double *optical_thickness(
    double *surface_density_profile, float opacity, int dim) {
  double *optical_thickness = malloc(dim*dim*sizeof(double));
  for ( int i = 0; i < dim*dim; ++i ) {
    optical_thickness[i] = 1.0-exp(-surface_density_profile[i]*opacity);
  }
  return optical_thickness;
}

double bb(double temperature, double wavelength, int dim) {
  double nu = c/wavelength;   // Hz
  return (2.0*h*pow(nu, 3)/c2)*(1.0/(exp(h*nu/(kb*temperature))-1.0));
}


double *intensity(
    double *temperature_profile, double wavelength, double pixel_size, int dim) {
  double *intensity = malloc(dim*dim*sizeof(double));
  for ( int i = 0; i < dim*dim; ++i ) {
      intensity[i] = bb(temperature_profile[i], wavelength, dim)*pow(pixel_size, 2)*bb_to_jy;
  }
  return intensity;
}


int main() {
  int dim = 4;
  float pixel_size = 0.1;
  float factor = dim*pixel_size;
  double *xx, *yy;
  double *x = linspace(-0.5, 0.5, dim, 1.0F);
  double *mesh_x = meshgrid(x, dim, 1);
  double *mesh_y = meshgrid(x, dim, 0);
  struct Grid grid1D = grid(dim, pixel_size, 0.5, 0.33, 0);
  double *r = radius(grid1D.xx, grid1D.yy, dim);
  double *temperature = temperature_power_law(r, 1500.0, 0.5, 0.5, dim);
  for ( int i = 0; i < dim*dim; ++i ) {
    printf("%f\n", mesh_y[i]);
  }
  return 0;
}
