#ifndef SPECTRAL_H
#define SPECTRAL_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>

namespace py = pybind11;

std::vector<double> constant_temperature(
    std::vector<double> &radius, double stellar_radius,
    float stellar_temperature, long long dim);

std::vector<double> temperature_power_law(
    std::vector<double> &radius, float inner_temp,
    float inner_radius, float q, long long dim);

std::vector<double> surface_density_profile(
    std::vector<double> &radius, float inner_radius,
    float inner_sigma, float p, long long dim);

std::vector<double> azimuthal_modulation(
    std::vector<double> &xx, std::vector<double> &yy,
    float a, float phi, long long dim);

std::vector<double> optical_thickness(
    std::vector<double> &surface_density_profile, double opacity, long long dim);

double bb(double temperature, double wavelength);

std::vector<double> intensity(
    std::vector<double> &temperature_profile,
    double wavelength, double pixel_size, long long dim);

std::vector<double> flat_disk(
    std::vector<double> &radius, std::vector<double> &xx,
    std::vector<double> &yy, double wavelength, double pixel_size,
    double stellar_radius, float stellar_temperature,
    float inner_temp, float inner_radius, float q, double opacity,
    float inner_sigma, float p, float a, double phi, long long dim,
    int modulated, int const_temperature);


py::array_t<double> constant_temperature(
        py::array_t<double> radius, double stellar_radius, float stellar_temperature) {
    py::buffer_info buf1 = radius.request();

    auto result = py::array_t<double>(buf1.size);
    double *ptr1 = static_cast<double *>(buf1.ptr);

    auto dim = static_cast<long long>(std::sqrt(dims));
    result = constant_temperature(radius, stellar_radius, stellar_temperature, dim);
}


PYBIND11_MODULE(spectral, mod) {
    mod.def("constant_temperature", &constant_temperature, "Recursive fibinacci algorithm.");
}

#endif
