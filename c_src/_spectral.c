#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include "spectral.h"


static char module_docstring[] =
    "This module contains functionality to calculate temperature gradient models";

static char linspace_docstring[] =
    "Calculates a linear grid.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "start : float\n"
    "   The start value.\n"
    "end : float\n"
    "   The end value.\n"
    "dim : int\n"
    "   The dimension of the grid.\n"
    "factor : float\n"
    "   The factor.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "linear_grid : astropy.units.mas";

static char meshgrid_docstring[] =
    "Calculates a meshgrid.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "linear_grid : astropy.units.mas\n"
    "   The grid.\n"
    "dim : int\n"
    "   The dimension of the grid.\n"
    "axis : int\n"
    "   The axis.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "mesh : astropy.units.mas";

static char grid_docstring[] =
    "Calculates the grid.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "dim : int\n"
    "   The dimension of the grid.\n"
    "pixel_size : float\n"
    "   The pixel size.\n"
    "pa : float\n"
    "   The position angle.\n"
    "elong : float\n"
    "   The ellipticity.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "xx : astropy.units.mas\n"
    "yy : astropy.units.mas\n";

static char radius_docstring[] =
    "Calculates the radius.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "xx : astropy.units.mas\n"
    "    The x-coordinate grid.\n"
    "yy : astropy.units.mas\n"
    "    The y-coordinate grid.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "radius : astropy.units.mas\n";

static char constant_temperature_docstring[] =
    "Calculates a constant temperature profile.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "radius : astroy.units.mas\n"
    "    The radial grid.\n"
    "stellar_radius : astropy.units.mas\n"
    "    The radius of the star.\n"
    "stellar_temperature : astropy.units.K\n"
    "    The effective temperature of the star.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "temperature_profile : astropy.units.K\n"
    "\n"
    "Notes\n"
    "-----\n"
    "The constant grey body profile is calculated in the follwing\n"
    "\n"
    ".. math:: R_* = \\sqrt{\\frac{L_*}{4\\pi\\sigma_sb\\T_*^4}}.\n"
    "\n"
    "And with this the individual grain's temperature profile is\n"
    "\n"
    ".. math:: T_{grain} = \\sqrt{\\frac{R_*}{2r}}\\cdot T_*.\n";

static char temperature_power_law_docstring[] =
    "Calculates a temperature power law profile.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "radius : astroy.units.mas\n"
    "    The radial grid.\n"
    "inner_temp : astropy.units.K\n"
    "    The temperature of the innermost grain.\n"
    "inner_radius : astropy.units.m\n"
    "    The radius of the innermost grain.\n"
    "q : astropy.units.one\n"
    "\n"
    "Returns\n"
    "-------\n"
    "temperature_profile : astropy.units.K\n"
    "\n"
    "Notes\n"
    "-----\n"
    "In case of a radial power law the formula is\n"
    "\n"
    ".. math:: T = T_0 * (1+\\frac{r}{R_0})^\\q.\n";

static char surface_density_profile_docstring[] =
    "Calculates the surface density profile.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "radius : astroy.units.mas\n"
    "    The radial grid.\n"
    "inner_sigma : astropy.units.g/astropy.units.cm**2\n"
    "    The temperature of the innermost grain.\n"
    "inner_radius : astropy.units.m\n"
    "    The radius of the innermost grain.\n"
    "q : astropy.units.one\n"
    "\n"
    "\n"
    "Returns\n"
    "-------\n"
    "surface_density_profile : astropy.units.g/astropy.units.cm**2\n";

static char azimuthal_modulation_docstring[] =
    "Calculates the azimuthal modulation.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "xx : astropy.units.mas\n"
    "    The x-coordinate grid.\n"
    "yy : astropy.units.mas\n"
    "    The y-coordinate grid.\n"
    "a : astropy.units.one\n"
    "    The amplitude of the modulation.\n"
    "phi : astropy.units.rad\n"
    "    The phase of the modulation.\n"
    "\n"
    "Returns\n"
    "-------\n"
    "azimuthal_modulation : astropy.units.one\n"
    "\n"
    "Notes\n"
    "-----\n"
    "Derived via trigonometry from Lazareff et al. 2017's:\n"
    "\n"
    "$ F(r) = F_{0}(r)\\cdot\\left(1+\\sum_{j=1}^{m}()c_{j}\\cos(j\\phi)+s_{j}\\sin(j\\phi)\\right)$\n";

static char optical_thickness_docstring[] =
    "Calculates the optical thickness.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "surface_density_profile : astropy.units.g/astropy.units.cm**2\n"
    "opacity : astropy.units.g/astropy.units.cm**2\n"
    "\n"
    "Returns\n"
    "-------\n"
    "optical_thickness : astropy.units.one\n";

static char bb_docstring[] =
    "Calculates the black body radiation.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "temperature : astropy.units.K\n"
    "wavelength : astropy.units.cm\n"
    "\n"
    "Returns\n"
    "-------\n"
    "intensity : astropy.units.erg/astropy.units.cm**2/astropy.units.sr/astropy.units.Hz\n";

static char intensity_docstring[] =
    "Calculates the intensity.\n"
    "\n"
    "Parameters\n"
    "----------\n"
    "temperature_profile : astropy.units.K\n"
    "wavelength : astropy.units.cm\n"
    "pixel_size : astropy.units.rad\n"
    "\n"
    "Returns\n"
    "-------\n"
    "intensity : astropy.units.Jy\n";
    
  
static PyObject *spectral_linspace(PyObject*self, PyObject *args);
static PyObject *spectral_meshgrid(PyObject*self, PyObject *args);
static PyObject *spectral_grid(PyObject*self, PyObject *args);
static PyObject *spectral_radius(PyObject*self, PyObject *args);
static PyObject *spectral_constant_temperature(PyObject*self, PyObject *args);
static PyObject *spectral_temperature_power_law(PyObject*self, PyObject *args);
static PyObject *spectral_surface_density_profile(PyObject*self, PyObject *args);
static PyObject *spectral_azimuthal_modulation(PyObject*self, PyObject *args);
static PyObject *spectral_optical_thickness(PyObject*self, PyObject *args);
static PyObject *spectral_bb(PyObject*self, PyObject *args);
static PyObject *spectral_intensity(PyObject*self, PyObject *args);


static PyMethodDef module_methods[] = {
    {"linspace", spectral_linspace, METH_VARARGS, linspace_docstring},
    {"meshgrid", spectral_meshgrid, METH_VARARGS, meshgrid_docstring},
    {"grid", spectral_grid, METH_VARARGS, grid_docstring},
    {"radius", spectral_radius, METH_VARARGS, radius_docstring},
    {"constant_temperature", spectral_constant_temperature, METH_VARARGS, constant_temperature_docstring},
    {"temperature_power_law", spectral_temperature_power_law, METH_VARARGS, temperature_power_law_docstring},
    {"surface_density_profile", spectral_surface_density_profile, METH_VARARGS, surface_density_profile_docstring},
    {"azimuthal_modulation", spectral_azimuthal_modulation, METH_VARARGS, azimuthal_modulation_docstring},
    {"optical_thickness", spectral_optical_thickness, METH_VARARGS, optical_thickness_docstring},
    {"bb", spectral_bb, METH_VARARGS, bb_docstring},
    {"intensity", spectral_intensity, METH_VARARGS, intensity_docstring},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC *PyInit__spectral(void)
{
    PyObject *module;
    static struct PyModuleDef mymodule = {
        PyModuleDef_HEAD_INIT,
        "spectral",
		module_docstring,
        -1,
		module_methods,
    };

    module = PyModule_Create(&mymodule); 
	if (module == NULL)
		return NULL;

	import_array();
    return module;
}


static PyObject *spectral_linspace(PyObject*self, PyObject *args)
{
    float start, stop;
    long long dim;
    double factor;

    if (!PyArg_ParseTuple(args, "ffLd", &start, &stop, &dim, &factor))
        return NULL;

    double *linear_grid = linspace(start, stop, dim, factor);
    PyObject *linear_grid_array = PyArray_SimpleNewFromData(1, &dim, NPY_DOUBLE, linear_grid);

    // Handle the NumPy array creation error
    if (linear_grid_array == NULL) {
        Py_XDECREF(linear_grid_array);
        free(linear_grid);
        return NULL;
    }

    return linear_grid_array;
}

static PyObject *spectral_meshgrid(PyObject*self, PyObject *args)
{
    double *linear_grid_obj;
    long int axis;

    if (!PyArg_ParseTuple(args, "Ol", &linear_grid_obj, &axis))
        return NULL;

    PyObject *linear_grid_array = PyArray_FROM_OTF(linear_grid_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    if (linear_grid_array == NULL) {
        Py_XDECREF(linear_grid_array);
       return NULL;
    }

    long long dim = (long long)PyArray_SIZE((PyArrayObject*)linear_grid_array);
    long long dims = dim*dim;

    double *linear_grid = (double*)PyArray_DATA(linear_grid_array);
    double *meshgrid_obj = meshgrid(linear_grid, dim, axis);
    PyObject *mesh_array = PyArray_SimpleNewFromData(1, &dims, NPY_DOUBLE, meshgrid_obj);
    PyObject *ret = Py_BuildValue("O", mesh_array);

    Py_XDECREF(linear_grid_array);
    Py_XDECREF(mesh_array);

    return ret;
}

static PyObject *spectral_grid(PyObject*self, PyObject *args)
{
    long long dim;
    int elliptic;
    float pixel_size, pa, elong;

    if (!PyArg_ParseTuple(args, "Lfffp", &dim, &pixel_size, &pa, &elong, &elliptic))
        return NULL;

    long long dims = dim*dim;

    struct Grid grid_obj = grid(dim, pixel_size, pa, elong, elliptic);
    PyObject *mesh_x_array = PyArray_SimpleNewFromData(1, &dims, NPY_DOUBLE, grid_obj.xx);
    PyObject *mesh_y_array = PyArray_SimpleNewFromData(1, &dims, NPY_DOUBLE, grid_obj.yy);
    PyObject *ret = Py_BuildValue("OO", mesh_x_array, mesh_y_array);

    Py_XDECREF(mesh_x_array);
    Py_XDECREF(mesh_y_array);

    return ret;
}

static PyObject *spectral_radius(PyObject*self, PyObject *args)
{
    double *xx_obj, *yy_obj;

    if (!PyArg_ParseTuple(args, "OO", &xx_obj, &yy_obj))
        return NULL;

    PyObject *xx_array = PyArray_FROM_OTF(xx_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *yy_array = PyArray_FROM_OTF(yy_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    if (xx_array == NULL || yy_array == NULL) {
        Py_XDECREF(xx_array);
        Py_XDECREF(yy_array);
        return NULL;
    }

    long long dims = (long long)PyArray_SIZE((PyArrayObject*)xx_array);
    long long dim = sqrt(dims);

    double *xx = (double*)PyArray_DATA(xx_array);
    double *yy = (double*)PyArray_DATA(yy_array);
    double *radius_obj = radius(xx, yy, dim);
    PyObject *radius_array = PyArray_SimpleNewFromData(1, &dims, NPY_DOUBLE, radius_obj);
    PyObject *ret = Py_BuildValue("O", radius_array);
        
    Py_XDECREF(xx_array);
    Py_XDECREF(yy_array);
    Py_XDECREF(radius_array);

    return ret;
}


static PyObject *spectral_constant_temperature(PyObject*self, PyObject *args)
{
    double *radius_obj;
    float stellar_radius, stellar_temperature;

    if (!PyArg_ParseTuple(args, "Off", &radius_obj, &stellar_radius, &stellar_temperature))
        return NULL;

    PyObject *radius_array = PyArray_FROM_OTF(radius_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    if (radius_array == NULL) {
        Py_XDECREF(radius_array);
        return NULL;
    }

    long long dims = (long long)PyArray_SIZE((PyArrayObject*)radius_array);
    long long dim = sqrt(dims);

    double *radius = (double*)PyArray_DATA(radius_array);
    double *const_temperature_obj = constant_temperature(radius, stellar_radius, stellar_temperature, dim);
    PyObject *const_temperature_array = PyArray_SimpleNewFromData(1, &dims, NPY_DOUBLE, const_temperature_obj);
    PyObject *ret = Py_BuildValue("O", const_temperature_array);

    Py_XDECREF(radius_array);
    Py_XDECREF(const_temperature_array);

    return ret;
}

static PyObject *spectral_temperature_power_law(PyObject*self, PyObject *args)
{
    double *radius_obj;
    float inner_temp, inner_radius, q;

    if (!PyArg_ParseTuple(args, "Offf", &radius_obj, &inner_temp, &inner_radius, &q))
        return NULL;

    PyObject *radius_array = PyArray_FROM_OTF(radius_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    if (radius_array == NULL) {
        Py_XDECREF(radius_array);
        return NULL;
    }

    long long dims = (long long)PyArray_SIZE((PyArrayObject*)radius_array);
    long long dim = sqrt(dims);

    double *radius = (double*)PyArray_DATA(radius_array);
    double *temperature_power_law_obj = temperature_power_law(radius, inner_temp, inner_radius, q, dim);
    PyObject *temperature_power_law_array = PyArray_SimpleNewFromData(1, &dims, NPY_DOUBLE, temperature_power_law_obj);
    PyObject *ret = Py_BuildValue("O", temperature_power_law_array);

    Py_XDECREF(radius_array);
    Py_XDECREF(temperature_power_law_array);

    return ret;
}

static PyObject *spectral_surface_density_profile(PyObject*self, PyObject *args)
{
    double *radius_obj;
    float inner_radius, inner_sigma, p;

    if (!PyArg_ParseTuple(args, "Offf", &radius_obj, &inner_radius, &inner_sigma, &p))
        return NULL;

    PyObject *radius_array = PyArray_FROM_OTF(radius_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    if (radius_array == NULL) {
        Py_XDECREF(radius_array);
        return NULL;
    }

    long long dims = (long long)PyArray_SIZE((PyArrayObject*)radius_array);
    long long dim = sqrt(dims);

    double *radius = (double*)PyArray_DATA(radius_array);
    double *sigma_profile_obj = surface_density_profile(radius, inner_radius, inner_sigma, p, dim);
    PyObject *sigma_profile_array = PyArray_SimpleNewFromData(1, &dims, NPY_DOUBLE, sigma_profile_obj);
    PyObject *ret = Py_BuildValue("O", sigma_profile_array);

    Py_XDECREF(radius_array);
    Py_XDECREF(sigma_profile_array);

    return ret;
}

static PyObject *spectral_azimuthal_modulation(PyObject*self, PyObject *args)
{
    double *xx_obj, *yy_obj;
    float a, phi;

    if (!PyArg_ParseTuple(args, "OOff", &xx_obj, &yy_obj, &a, &phi))
        return NULL;

    PyObject *xx_array = PyArray_FROM_OTF(xx_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *yy_array = PyArray_FROM_OTF(yy_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    if (xx_array == NULL || yy_array == NULL) {
        Py_XDECREF(xx_array);
        Py_XDECREF(yy_array);
        return NULL;
    }

    long long dims = (long long)PyArray_SIZE((PyArrayObject*)xx_array);
    long long dim = sqrt(dims);

    double *xx = (double*)PyArray_DATA(xx_array);
    double *yy = (double*)PyArray_DATA(yy_array);
    double *azimuthal_modulation_obj = azimuthal_modulation(xx, yy, a, phi, dim);
    PyObject *azimuthal_modulation_array = PyArray_SimpleNewFromData(1, &dims, NPY_DOUBLE, azimuthal_modulation_obj);
    PyObject *ret = Py_BuildValue("O", azimuthal_modulation_array);

    Py_XDECREF(xx_array);
    Py_XDECREF(yy_array);
    Py_XDECREF(azimuthal_modulation_array);

    return ret;
}

static PyObject *spectral_optical_thickness(PyObject*self, PyObject *args)
{
    double *surface_density_profile_obj;
    float opacity;

    if (!PyArg_ParseTuple(args, "Of", &surface_density_profile_obj, &opacity))
        return NULL;

    PyObject *surface_density_profile_array = PyArray_FROM_OTF(surface_density_profile_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    if (surface_density_profile_array == NULL) {
        Py_XDECREF(surface_density_profile_array);
        return NULL;
    }

    long long dims = (long long)PyArray_SIZE((PyArrayObject*)surface_density_profile_array);
    long long dim = sqrt(dims);

    double *surface_density_profile = (double*)PyArray_DATA(surface_density_profile_array);
    double *optical_thickness_obj = optical_thickness(surface_density_profile, opacity, dim);
    PyObject *optical_thickness_array = PyArray_SimpleNewFromData(1, &dims, NPY_DOUBLE, optical_thickness_obj);
    PyObject *ret = Py_BuildValue("O", optical_thickness_array);

    Py_XDECREF(surface_density_profile_array);
    Py_XDECREF(optical_thickness_array);

    return ret;
}

static PyObject *spectral_bb(PyObject*self, PyObject *args)
{
    double temperature, wavelength;

    if (!PyArg_ParseTuple(args, "dd", &temperature, &wavelength))
        return NULL;

    double bb_obj = bb(temperature, wavelength);
    PyObject *ret = Py_BuildValue("d", bb_obj);
    return ret;
}

static PyObject *spectral_intensity(PyObject*self, PyObject *args)
{
    double *temperature_profile_obj, wavelength, pixel_size;

    if (!PyArg_ParseTuple(args, "Odd", &temperature_profile_obj, &wavelength, &pixel_size))
        return NULL;

    PyObject *temperature_profile_array = PyArray_FROM_OTF(temperature_profile_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    if ( temperature_profile_array == NULL) {
        Py_XDECREF(temperature_profile_array);
        return NULL;
    }

    long long dims = (long long)PyArray_SIZE((PyArrayObject*)temperature_profile_array);
    long long dim = sqrt(dims);

    double *temperature_profile = (double*)PyArray_DATA(temperature_profile_array);

    double *intensity_obj = intensity(temperature_profile, wavelength, pixel_size, dim);
    PyObject *intensity_array = PyArray_SimpleNewFromData(1, &dims, NPY_DOUBLE, intensity_obj);
    PyObject *ret = Py_BuildValue("O", intensity_array);

    Py_XDECREF(temperature_profile_array);
    Py_XDECREF(intensity_array);
            
    return ret;
}
