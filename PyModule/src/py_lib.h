#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace kerbal_guidance_system
{
    //static py::tuple py_simulate_atm_phase(double t, py::array_t<double> py_yi, py::list py_events, py::dict py_params);
    // static py::tuple py_simulate_vac_phase(double t, py::array_t<double> py_y, py::list py_events, py::dict py_params);
    // py::array_t<double> kgs_constraint_residuals(double t, py::array_t<double> py_y, py::list py_events, py::dict py_p);
    py::array_t<double> py_output_atm_phase(double t, py::array_t<double> py_y, py::list py_events, py::dict py_params);
    // py::tuple test(py::array_t<double> arr);
}