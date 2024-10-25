#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace kerbal_guidance_system
{
	py::tuple py_simulate_atm_phase(double t, py::array_t<double> py_yi, double pitch_rate, double azimuth, py::list py_events, py::dict py_params);
	py::array_t<double> py_output_atm_phase(double t, py::array_t<double> py_yi, double pitch_rate, double azimuth, py::list py_events, py::dict py_params);
	py::tuple py_simulate_vac_phase(double t, py::array_t<double> py_yi, py::array_t<double> py_x, py::list py_events, py::dict py_params);
	py::tuple py_simulate_vac_phase_to_velocity(double t, py::array_t<double> py_yi, py::array_t<double> py_x, double final_velocity, py::list py_events, py::dict py_params);
	py::array_t<double> py_output_vac_phase(double t, py::array_t<double> py_yi, py::array_t<double> py_x, py::list py_events, py::dict py_params);
	py::array_t<double> py_constraint_residuals(double t, py::array_t<double> py_yi, py::array_t<double> py_x, py::array_t<double> py_c, py::list py_events, py::dict py_params);
	py::array_t<double> py_constraint_jacobian(double t, py::array_t<double> py_yi, py::array_t<double> py_x, py::array_t<double> py_c, py::list py_events, py::dict py_params);
}