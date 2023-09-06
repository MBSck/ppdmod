#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <spectral.h>

namespace py = pybind11;

py::array_t<double> py_constant_temperature(
  py::array_t<double, py::array::c_style | py::array::forcecast> radius,
  double stellar_radius, float stellar_temperature) {
    std::vector<double> radius_vec(radius.size());

    std::memcpy(radius_vec.data(), radius.data(), radius.size()*sizeof(double));
    std::vector<double> result_vec = constant_temperature(radius_vec, stellar_radius, stellar_temperature, radius.size());
    auto result = py::array_t<double>(radius.size());
    auto result_buffer = result.request();
    int *result_ptr = static_cast<int*>(result_buffer.ptr);

    std::memcpy(result_ptr, result_vec.data(), result_vec.size()*sizeof(double));

    return result;
}

py::array_t<double> py_temperature_power_law(
  py::array_t<double, py::array::c_style | py::array::forcecast> radius,
  float inner_radius, float inner_temp, float q, long long dim) {
    std::vector<double> radius_vec(radius.size());

    std::memcpy(radius_vec.data(), radius.data(), radius.size()*sizeof(double));
    std::vector<double> result_vec = temperature_power_law(radius_vec, inner_radius, inner_temp, q, radius.size());
    auto result = py::array_t<double>(radius.size());
    auto result_buffer = result.request();
    int *result_ptr = static_cast<int*>(result_buffer.ptr);

    std::memcpy(result_ptr, result_vec.data(), result_vec.size()*sizeof(double));

    return result;
}

py::array_t<double> py_surface_density_profile(
  py::array_t<double, py::array::c_style | py::array::forcecast> radius,
  float inner_radius, float inner_sigma, float p, long long dim) {
    std::vector<double> radius_vec(radius.size());

    std::memcpy(radius_vec.data(), radius.data(), radius.size()*sizeof(double));
    std::vector<double> result_vec = surface_density_profile(radius_vec, inner_radius, inner_sigma, p, radius.size());
    auto result = py::array_t<double>(radius.size());
    auto result_buffer = result.request();
    int *result_ptr = static_cast<int*>(result_buffer.ptr);

    std::memcpy(result_ptr, result_vec.data(), result_vec.size()*sizeof(double));

    return result;
}

py::array_t<double> py_azimuthal_modulation(
  py::array_t<double, py::array::c_style | py::array::forcecast> xx,
  py::array_t<double, py::array::c_style | py::array::forcecast> yy,
  float a, double phi, long long dim) {
    std::vector<double> xx_vec(xx.size());
    std::vector<double> yy_vec(yy.size());

    std::memcpy(xx_vec.data(), xx.data(), xx.size()*sizeof(double));
    std::memcpy(yy_vec.data(), yy.data(), yy.size()*sizeof(double));
    std::vector<double> result_vec = azimuthal_modulation(xx_vec, yy_vec, a, phi, xx.size());
    auto result = py::array_t<double>(xx.size());
    auto result_buffer = result.request();
    int *result_ptr = static_cast<int*>(result_buffer.ptr);

    std::memcpy(result_ptr, result_vec.data(), result_vec.size()*sizeof(double));

    return result;
}

py::array_t<double> py_optickal_thickness(
  py::array_t<double, py::array::c_style | py::array::forcecast> surface_density,
  double opacity, long long dim) {
    std::vector<double> surface_density_vec(surface_density.size());

    std::memcpy(surface_density_vec.data(), surface_density.data(), surface_density.size()*sizeof(double));
    std::vector<double> result_vec = optical_thickness(surface_density_vec, opacity, surface_density.size());
    auto result = py::array_t<double>(surface_density.size());
    auto result_buffer = result.request();
    int *result_ptr = static_cast<int*>(result_buffer.ptr);

    std::memcpy(result_ptr, result_vec.data(), result_vec.size()*sizeof(double));

    return result;
}

py::array_t<double> py_intensity(
  py::array_t<double, py::array::c_style | py::array::forcecast> temperature,
  double wavelength, double pixel_size, long long dim) {
    std::vector<double> temperature_vec(temperature.size());

    std::memcpy(temperature_vec.data(), temperature.data(), temperature.size()*sizeof(double));
    std::vector<double> result_vec = intensity(temperature_vec, wavelength, pixel_size, temperature.size());
    auto result = py::array_t<double>(temperature.size());
    auto result_buffer = result.request();
    int *result_ptr = static_cast<int*>(result_buffer.ptr);

    std::memcpy(result_ptr, result_vec.data(), result_vec.size()*sizeof(double));

    return result;
}

PYBIND11_MODULE(_spectral, mod) {
    mod.def("constant_temperature", &py_constant_temperature, "Test");
    mod.def("temperature_power_law", &py_temperature_power_law, "Test");
    mod.def("surface_density_profile", &py_surface_density_profile, "Test");
    mod.def("azimuthal_modulation", &py_azimuthal_modulation, "Test");
    mod.def("optical_thickness", &py_optickal_thickness, "Test");
    mod.def("intensity", &py_intensity, "Test");
}

