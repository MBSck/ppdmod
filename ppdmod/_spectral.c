#include <Python.h>
#include <numpy/arrayobject.h>
#include "spectral.h"


static char module_docstring[] =
    "This module contains functionality to calculate temperature gradient models";

static PyObject *linespace(PyObject*self, PyObject *args);
static PyObject *meshgrid(PyObject*self, PyObject *args);
static PyObject *grid(PyObject*self, PyObject *args);
static PyObject *radius(PyObject*self, PyObject *args);
static PyObject *constant_temperature(PyObject*self, PyObject *args);
static PyObject *temperature_power_law(PyObject*self, PyObject *args);
static PyObject *surface_density_profile(PyObject*self, PyObject *args);
static PyObject *azimuthal_modulation(PyObject*self, PyObject *args);
static PyObject *optical_thickness(PyObject*self, PyObject *args);
static PyObject *bb(PyObject*self, PyObject *args);
static PyObject *intensity(PyObject*self, PyObject *args);

static PyMethodDef module_methods[] = {
    {"set_linspace", linspace, METH_VARARGS, linspace_docstring},
    {"create_meshgrid", meshgrid, METH_VARARGS, meshgrid_docstring},
    {"grid", grid, METH_VARARGS, grid_docstring},
    {"radius", radius, METH_VARARGS, radius_docstring},
    {"constant_temperature", constant_temperature, METH_VARARGS, constant_temperature_docstring},
    {"temperature_power_law", temperature_power_law, METH_VARARGS, temperature_power_law_docstring},
    {"surface_density_profile", surface_density_profile, METH_VARARGS, surface_density_profile_docstring},
    {"azimuthal_modulation", azimuthal_modulation, METH_VARARGS, azimuthal_modulation_docstring},
    {"optical_thickness", optical_thickness, METH_VARARGS, optical_thickness_docstring},
    {"bb", bb, METH_VARARGS, bb_docstring},
    {"intensity", intensity, METH_VARARGS, intensity_docstring},
};
