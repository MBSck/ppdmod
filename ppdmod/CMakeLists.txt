cmake_minimum_required(VERSION 3.5)

project(spectral LANGUAGES C VERSION 1.0)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(PYTHON_INCLUDE_DIRS "/Users/scheuck/.pyenv/versions/3.11.3/include/python3.11")
set(PYTHON_LIBRARIES "/Users/scheuck/.pyenv/versions/3.11.3/lib")
set(NUMPY_INCLUDE_DIRS "/Users/scheuck/.pyenv/versions/3.11.3/lib/python3.11/site-packages/numpy/core/include/")
set(NUMPY_LIBRARIES "/Users/scheuck/.pyenv/versions/3.11.3/lib/python3.11/site-packages/numpy/core/lib")
include_directories(${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIRS})
add_executable(spectral spectral.c ${PYTHON_LIBRARIES} ${NUMPY_LIBRARIES})
