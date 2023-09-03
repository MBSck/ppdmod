cmake_minimum_required(VERSION 3.5)

project(spectral VERSION 1.0)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add your source files
find_package(Boost REQUIRED COMPONENTS python39)
add_executable(spectral spectral.cpp)
target_link_libraries(spectral ${Boost_LIBRARIES})
