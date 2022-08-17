from setuptools import setup, find_packages


with open("README.md", "r",) as fh:
    long_description = fh.read()

setup(
    name="ppdmod",
    version="0.1",
    description="PPDMod is a pipeline to model and fit protoplanetary disk"\
                " data observed with the MATISSE-pipeline",
    author="Marten Scheuck",
    url="",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: GNU General Public License v2 or later \
        (GPLv2+)",
        "Operating System :: OS Independent",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",

    # Fo production dependencies, versions should be as relaxed as possible
    install_requires = [],

    # Dev requires used for developing this package -> Better than
    # requirements.txt. For optional requirements, versions should be specific
    extras_require = {
        "dev": [
            "pytest>=3.7",
            "numpy",
            "scipy",
            "emcee",
            "dynesty",
            "matplotlib"
        ],
    },
    # requirements.txt is for apps deployed on machines you control
    # Fixed version numbers, e.g., requests==2.22.0
    # Is generated with pip freeze > requirements.txt
)

