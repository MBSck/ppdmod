from setuptools import setup, Extension

import numpy as np
from Cython.Compiler import Options

Options.docstrings = True
Options.annotate = True
Options.language_level = 3
Options.profile = False


extensions = [Extension("_spectral_c", ["c_src/_spectral.c", "c_src/spectral.c"]),
              Extension("_spectral_cy", ["cython_src/spectral.pyx"],)]

setup(
    name="ppdmod",
    ext_modules=extensions,
    include_dirs=[np.get_include()]
)
