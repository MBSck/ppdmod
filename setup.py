from setuptools import setup

import numpy as np
from Cython.Build import cythonize
from Cython.Compiler import Options

Options.docstrings = True
Options.annotate = False

setup(
    name="ppdmod",
    ext_modules=cythonize("ppdmod/spectral.pyx"),
    include_dirs=[np.get_include()]
)
