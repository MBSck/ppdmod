#include <stdio.h>

double* set_linspace(
    float start, float end, int dim,
    float factor = 1.0) {
  int grid_length = dim;
  double step = (end - start)/dim;
  double* grid = new double[dim];
  for ( int i = 0; i < dim; i++ )
    grid[i] = (start + i * step) * factor;
  return grid;
}


double* set_meshgrid(double* grid, int dim, int axis = 0) {
  double* mesh = new double[dim*dim];
  double temp;
  for ( int i = 0; i < dim; i++ ) {
      for ( int j = 0; j < dim; j++ ) {
        if (axis == 0) {
          temp = grid[i];
        } else {
          temp = grid[j];
        }
        mesh[i*10+j] = temp;
      }
    }
  return mesh;
}


int main() {
  int dim = 10;
  float factor = dim*0.1;
  double* linspace = set_linspace(-0.5, 0.5, dim, factor);
  double* mesh_y = set_meshgrid(linspace, dim, 0);
  return 0;
}
