from setuptools import setup

from pybind11.setup_helpers import Pybind11Extension, build_ext


spectral_cpp = Pybind11Extension(
    "ppdmod._spectral",
    ["src/spectral.cpp", "src/spectral_wrapper.cpp"],
    include_dirs=["include/"],
    extra_compile_args=["-O3", "-march=native", "-fno-math-errno"])

setup(
    name="ppdmod",
    version="0.9.0",
    author="Marten Scheuck",
    author_email="martenscheuck@gmail.com",
    url="https://github.com/MBSck/ppdmod",
    packages=["ppdmod"],
    # readme="README.md",
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
    ext_modules=[spectral_cpp],
    cmdclass={"build_ext": build_ext},
)
