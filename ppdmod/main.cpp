#include <iostream>
#include <tuple>



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

std::tuple<double*, double*> calculate_grid(int dim, float pixel_size) {
  double* linspace = set_linspace(-0.5, 0.5, dim, dim*pixel_size);
  double* mesh_x = set_meshgrid(linspace, dim, 1);
  double* mesh_y = set_meshgrid(linspace, dim, 0);
  return std::make_tuple(mesh_x, mesh_y);
}

int main() {
  int dim = 10;
  float pixel_size = 0.1;
  float factor = dim*pixel_size;
  double* xx, *yy;
  std::tie(xx, yy) = calculate_grid(dim, pixel_size);
  for ( int i = 0; i < dim*dim; i++ )
    printf("%f\n", yy[i]);
  return 0;
}
