#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "moving_average.h"

namespace py = pybind11;

PYBIND11_MODULE(moving_average, m) {
    m.def("moving_average", &moving_average, 
          "Compute the moving average of a list of numbers", 
          py::arg("data"), py::arg("period"));
}
