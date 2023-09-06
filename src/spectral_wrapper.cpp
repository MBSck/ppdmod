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
    int *result_ptr    = static_cast<int*>(result_buffer.ptr);

    std::memcpy(result_ptr,result_vec.data(),result_vec.size()*sizeof(double));

    return result;
}


PYBIND11_MODULE(spectral, mod) {
    mod.def("constant_temperature", &py_constant_temperature, "Test");
}

