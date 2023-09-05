from setuptools import setup, Extension

import numpy as np
from Cython.Compiler import Options

Options.docstrings = True
Options.annotate = True
Options.language_level = 3
Options.profile = False


extensions = [Extension("ppdmod._spectral", ["c_src/_spectral.c", "c_src/spectral.c"]),
              Extension("ppdmod._spectral_cy", ["cython_src/spectral.pyx"],)]

setup(
    name="ppdmod",
    version="0.9.0",
    author="Marten Scheuck",
    author_email="martenscheuck@gmail.com",
    url="https://github.com/MBSck/ppdmod",
    packages=["ppdmod"],
    readme="README.md",
    description="A package for modelling and model-fitting protoplanetary disks",
    classifiers=[
        "Development Status :: 1 - Planning",
        "Framework :: Pytest",
        "Framework :: Sphinx",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    include_dirs=[np.get_include()],
    install_requires=[
        'importlib-metadata; python_version<"3.8"',
        "astropy >=5.1",
        "corner >=2.2.2",
        "emcee >=3.1.2",
        "matplotlib >=3.5.3",
        "numpy >=1.23.2",
        "pip >=22.2.2",
        "scipy >=1.9.1",
        "tqdm >=4.64.0",
        "pyfftw >=0.13.0",
        "openpyxl >= 3.1.2",
        "cython >= 3.0.2",
        "pandas >= 2.1.0",
    ],
    ext_modules=extensions,
)
