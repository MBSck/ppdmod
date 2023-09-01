from setuptools import setup

import numpy as np
from Cython.Build import cythonize
from Cython.Compiler import Options

Options.docstrings = True
Options.annotate = True


setup(
    name="ppdmod",
    ext_modules=cythonize("ppdmod/spectral.pyx",
                          compiler_directives={"language_level": 3,
                                               "profile": False}),
    include_dirs=[np.get_include()]
)
