from setuptools import setup, find_packages

with open("README.md", "r",) as fh:
    long_description = fh.read()

setup(
    name="ppdmodler",
    version="0.0.1",
    description="Modelling tools for protoplanetary disks",
    author="Marten Scheuck",
    url="",
    py_modules=["main"],
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
        ],
    },
    # requirements.txt is for apps deployed on machines you control
    # Fixed version numbers, e.g., requests==2.22.0
    # Is generated with pip freeze > requirements.txt
)

