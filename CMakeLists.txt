cmake_minimum_required(VERSION 3.5)
project(spectral)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(PROJECT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(PROJECT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/spectral.cpp
    ${PROJECT_SOURCE_DIR}/spectral_wrapper.cpp
)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
include_directories(${PROJECT_INCLUDE_DIR} ${Python3_INCLUDE_DIRS})

find_package(pybind11 REQUIRED)
add_library(spectral_lib SHARED ${SOURCE_FILES})
pybind11_add_module(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PRIVATE spectral_lib ${Python3_LIBRARIES})
