#ifndef SPECTRAL_H
#define SPECTRAL_H


struct Grid;

double *set_linspace(float start, float end, int dim, double factor);

double *create_meshgrid(double *grid, int dim, int axis);

struct Grid calculate_grid(
    int dim, float pixel_size, float pa, float elong, int elliptic);

double *calculate_radius(double *xx, double *yy, int dim);

double *calculate_const_temperature(
    double *radius, float stellar_radius, float stellar_temperature, int dim);

double *calculate_temperature_power_law(
    double *radius, float inner_temp, float inner_radius, float q, int dim);

double *calculate_surface_density_profile(
    double *radius, float inner_radius,
    float inner_sigma, float p, int dim);

double *calculate_azimuthal_modulation(
    double *xx, double *yy, double a, double phi, int dim);

double *calculate_optical_thickness(
    double *surface_density_profile, float opacity, int dim);

double *calculate_intensity(
    double *temperature_profile, double wavelength, double pixel_size, int dim);

static char set_linspace_docstring[] =
    "";

static PyObject *spectral_set_linspace(PyObject *self, PyObject *args);

static char module_docstring[] =
    "";

// static PyMethodDef module_methods[] = {
//     {"chi2", spectral_set_linspace, METH_VARARGS | METH_KEYWORDS, set_linspace_docstring},
//     {NULL, NULL, 0, NULL}
// };
//
// static PyModuleDef mymodule = {
//     PyModuleDef_HEAD_INIT,
//     "_spectral",      // Module name
//     module_docstring,  // Module docstring
//     -1,           // Module state size, -1 means global variables are supported
//     module_methods   // Method table
// };
//
// PyMODINIT_FUNC init_spectral(void)
// {
//     PyObject *m = PyModule_Create(&mymodule);
//
//     /* Load `numpy` functionality. */
//     import_array();
// }



#endif
