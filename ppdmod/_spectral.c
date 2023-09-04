    30 }
#include <numpy/arrayobject.h>
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
    "dim : int"
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
    "dim : int"
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
    "dim : int"
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
    "dim : int"
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
    "dim : int"
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
    "dim : int"
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
    "dim : int"
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
    "dim : int"
    ""
    "Returns"
    "-------"
    "intensity : astropy.units.Jy";
    
  

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


PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
PyInit__web(void)
#else
init_web(void)
#endif
{
	#if PY_MAJOR_VERSION >= 3
		PyObject *module;
		static struct PyModuleDef moduledef = {
			PyModuleDef_HEAD_INIT,
			"_spectral",             /* m_name */
			module_docstring,    /* m_doc */
			-1,                  /* m_size */
			module_methods,      /* m_methods */
			NULL,                /* m_reload */
			NULL,                /* m_traverse */
			NULL,                /* m_clear */
			NULL,                /* m_free */
		};
	#endif

	#if PY_MAJOR_VERSION >= 3
		module = PyModule_Create(&moduledef);
		if (!module)
			return NULL;
		/* Load `numpy` functionality. */
		import_array();
		return module;
	#else
	    PyObject *m = Py_InitModule3("_spectral", module_methods, module_docstring);
		if (m == NULL)
			return;
		/* Load `numpy` functionality. */
		import_array();
	#endif
}
