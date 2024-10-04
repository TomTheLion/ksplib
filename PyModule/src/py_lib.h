#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace kerbal_guidance_system
{
    py::tuple simulate_atm_phase(double t, py::array_t<double> py_yi, py::list py_events, py::dict py_p);
    // py::tuple kgs_simulate_vac_phase(double t, py::array_t<double> py_y, py::list py_events, py::dict py_p);
    // py::array_t<double> kgs_constraint_residuals(double t, py::array_t<double> py_y, py::list py_events, py::dict py_p);
    // py::array_t<double> kgs_output_time_series(double t, py::array_t<double> py_y, py::list py_events, py::dict py_p_atm, py::dict py_p_vac);
    // py::tuple test();
}