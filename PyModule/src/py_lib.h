#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <Eigen/Dense>

namespace py = pybind11;
using namespace pybind11::literals;

namespace py_util
{
	struct Spl {
		double* t;
		double* c;
		size_t n;
		size_t k;
		Spl() : t(nullptr), c(nullptr), n(0), k(0) {};
		Spl(double* t, double* c, size_t n, size_t k) : t(t), c(c), n(n), k(k) {};
		double eval(double x) const;
	};
	size_t array_size(py::object obj);
	double* array_ptr(py::object obj);
	void array_copy(py::array_t<double> src, py::array_t<double> dst);
	void array_copy(std::vector<double> src, py::array_t<double> dst);
	Eigen::Vector3d vector3d(py::object obj);
	Eigen::Vector3d vector3d(py::object obj, int n);
	Spl spline(py::object obj);
}

namespace kerbal_guidance_system
{
	py::tuple py_simulate_atm_phase(double t, py::array_t<double> py_yi, double pitch_rate, double azimuth, py::list py_events, py::dict py_params);
	py::array_t<double> py_output_atm_phase(double t, py::array_t<double> py_yi, double pitch_rate, double azimuth, py::list py_events, py::dict py_params);
	py::tuple py_guidance_residuals_jacobian(double t, double coast_duration, py::array_t<double> py_yi, py::array_t<double> py_x, py::array_t<double> py_c, py::list py_events, py::dict py_params);
	py::tuple py_simulate_vac_phase_to_velocity(double t, py::array_t<double> py_yi, py::array_t<double> py_x, double final_velocity, py::list py_events, py::dict py_params);
	py::array_t<double> py_output_vac_phase(double t, py::array_t<double> py_yi, py::array_t<double> py_x, py::list py_events, py::dict py_params);
	py::array_t<double> py_constraint_residuals(double t, py::array_t<double> py_yi, py::array_t<double> py_x, py::array_t<double> py_c, py::list py_events, py::dict py_params);
	py::array_t<double> py_constraint_jacobian(double t, py::array_t<double> py_yi, py::array_t<double> py_x, py::array_t<double> py_c, py::list py_events, py::dict py_params);
}