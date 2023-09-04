#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include "spectral.h"


static char module_docstring[] =
    "This module contains functionality to calculate temperature gradient models";

static char linspace_docstring[] =
    "Calculates a linear grid."
    ""
    "Parameters"
    "----------"
    "start : float"
    "    The start value."
    "end : float"
    "    The end value."
    "dim : int"
    "    The dimension of the grid."
    "factor : float"
    "    The factor."
    ""
    "Returns"
    "-------"
    "linear_grid : astropy.units.mas"; ;

static char meshgrid_docstring[] =
    "Calculates a meshgrid."
    ""
    "Parameters"
    "----------"
    "linear_grid : astropy.units.mas"
    "    The grid."
    "dim : int"
    "    The dimension of the grid."
    "axis : int"
    "    The axis."
    ""
    "Returns"
    "-------"
    "mesh : astropy.units.mas";

static char grid_docstring[] =
    "Calculates the grid."
    ""
    "Parameters"
    "----------"
    "dim : int"
    "    The dimension of the grid."
    "pixel_size : float"
    "    The pixel size."
    "pa : float"
    "    The position angle."
    "elong : float"
    "    The ellipticity."
    ""
    "Returns"
    "-------"
    "xx : astropy.units.mas"
    "yy : astropy.units.mas";

static char radius_docstring[] =
    "Calculates the radius."
    ""
    "Parameters"
    "----------"
    "xx : astropy.units.mas"
    "    The x-coordinate grid."
    "yy : astropy.units.mas"
    "    The y-coordinate grid."
    ""
    "Returns"
    "-------"
    "radius : astropy.units.mas";

static char constant_temperature_docstring[] =
    "Calculates a constant temperature profile."
    ""
    "Parameters"
    "----------"
    "radius : astroy.units.mas"
    "    The radial grid."
    "stellar_radius : astropy.units.mas"
    "    The radius of the star."
    "stellar_temperature : astropy.units.K"
    "    The effective temperature of the star."
    ""
    "Returns"
    "-------"
    "temperature_profile : astropy.units.K"
    ""
    "Notes"
    "-----"
    "The constant grey body profile is calculated in the follwing"
    ""
    ".. math:: R_* = \\sqrt{\\frac{L_*}{4\\pi\\sigma_sb\\T_*^4}}."
    ""
    "And with this the individual grain's temperature profile is"
    ""
    ".. math:: T_{grain} = \\sqrt{\\frac{R_*}{2r}}\\cdot T_*.";

static char temperature_power_law_docstring[] =
    "Calculates a temperature power law profile."
    ""
    "Parameters"
    "----------"
    "radius : astroy.units.mas"
    "    The radial grid."
    "inner_temp : astropy.units.K"
    "    The temperature of the innermost grain."
    "inner_radius : astropy.units.m"
    "    The radius of the innermost grain."
    "q : astropy.units.one"
    ""
    "Returns"
    "-------"
    "temperature_profile : astropy.units.K"
    ""
    "Notes"
    "-----"
    "In case of a radial power law the formula is"
    ""
    ".. math:: T = T_0 * (1+\\frac{r}{R_0})^\\q.";

static char surface_density_profile_docstring[] =
    "Calculates the surface density profile."
    ""
    "Parameters"
    "----------"
    "radius : astroy.units.mas"
    "    The radial grid."
    ""
    "Returns"
    "-------"
    "surface_density_profile : astropy.units.g/astropy.units.cm**2";

static char azimuthal_modulation_docstring[] =
    "Calculates the azimuthal modulation."
    ""
    "Parameters"
    "----------"
    "xx : astropy.units.mas"
    "    The x-coordinate grid."
    "yy : astropy.units.mas"
    "    The y-coordinate grid."
    "a : astropy.units.one"
    "    The amplitude of the modulation."
    "phi : astropy.units.rad"
    "    The phase of the modulation."
    ""
    "Returns"
    "-------"
    "azimuthal_modulation : astropy.units.one"
    ""
    "Notes"
    "-----"
    "Derived via trigonometry from Lazareff et al. 2017's:"
    ""
    "$ F(r) = F_{0}(r)\\cdot\\left(1+\\sum_{j=1}^{m}()c_{j}\\cos(j\\phi)+s_{j}\\sin(j\\phi)\\right)$";

static char optical_thickness_docstring[] =
    "Calculates the optical thickness."
    ""
    "Parameters"
    "----------"
    "surface_density_profile : astropy.units.g/astropy.units.cm**2"
    "opacity : astropy.units.g/astropy.units.cm**2"
    ""
    "Returns"
    "-------"
    "optical_thickness : astropy.units.one";

static char bb_docstring[] =
    "Calculates the black body radiation."
    ""
    "Parameters"
    "----------"
    "temperature : astropy.units.K"
    "wavelength : astropy.units.cm"
    ""
    "Returns"
    "-------"
    "intensity : astropy.units.erg/astropy.units.cm**2/astropy.units.sr/astropy.units.Hz";

static char intensity_docstring[] =
    "Calculates the intensity."
    ""
    "Parameters"
    "----------"
    "temperature_profile : astropy.units.K"
    "wavelength : astropy.units.cm"
    "pixel_size : astropy.units.rad"
    ""
    "Returns"
    "-------"
    "intensity : astropy.units.Jy";
    
  
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
    long int dim;
    double factor;

    if (!PyArg_ParseTuple(args, "ffld", &start, &stop, &dim, &factor))
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

    if (!PyArg_ParseTuple(args, "Oi", &linear_grid_obj, &axis))
        return NULL;

    PyObject *linear_grid_array = PyArray_FROM_OTF(linear_grid_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    if (linear_grid_array == NULL) {
        Py_XDECREF(linear_grid_array);
        return NULL;
    }

    long int dim = (int)PyArray_SIZE((PyArrayObject*)linear_grid_array);
    long int dims = dim*dim;

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
    int dim, elliptic;
    float pixel_size, pa, elong;

    if (!PyArg_ParseTuple(args, "ifffp", &dim, &pixel_size, &pa, &elong, &elliptic))
        return NULL;

    int dims = dim*dim;

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

    long int dim = (int)PyArray_SIZE((PyArrayObject*)xx_array);
    long int dims = dim*dim;

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

    long int dim = (int)PyArray_SIZE((PyArrayObject*)radius_array);
    long int dims = dim*dim;

    double *radius = (double*)PyArray_DATA(radius_array);
    double *const_temperature_obj = const_temperature(radius, stellar_radius, stellar_temperature, dim);
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

    long int dim = (int)PyArray_SIZE((PyArrayObject*)radius_array);
    long int dims = dim*dim;

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

    long int dim = (int)PyArray_SIZE((PyArrayObject*)radius_array);
    long int dims = dim*dim;

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

    long int dim = (int)PyArray_SIZE((PyArrayObject*)xx_array);
    long int dims = dim*dim;

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

    if (!PyArg_ParseTuple(args, "Off", &surface_density_profile_obj, &opacity))
        return NULL;

    PyObject *surface_density_profile_array = PyArray_FROM_OTF(surface_density_profile_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    if (surface_density_profile_array == NULL) {
        Py_XDECREF(surface_density_profile_array);
        return NULL;
    }

    long int dim = (int)PyArray_SIZE((PyArrayObject*)surface_density_profile_array);
    long int dims = dim*dim;

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

    long int dim = (int)PyArray_SIZE((PyArrayObject*)temperature_profile_array);
    long int dims = dim*dim;

    double *temperature_profile = (double*)PyArray_DATA(temperature_profile_array);

    double *intensity_obj = intensity(temperature_profile, wavelength, pixel_size, dim);
    PyObject *intensity_array = PyArray_SimpleNewFromData(1, &dims, NPY_DOUBLE, intensity_obj);
    PyObject *ret = Py_BuildValue("O", intensity_array);

    Py_XDECREF(temperature_profile_array);
    Py_XDECREF(intensity_array);
            
    return ret;
}
