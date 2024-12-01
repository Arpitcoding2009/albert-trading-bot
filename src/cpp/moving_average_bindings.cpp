#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "moving_average.h"

namespace py = pybind11;

PYBIND11_MODULE(moving_average_module, m) {
    m.doc() = "High-Performance Moving Average Calculation";
    m.def("moving_average", &moving_average, 
          "Compute moving average of a vector", 
          py::arg("data"), py::arg("window"));
}
