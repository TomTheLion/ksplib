#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace pybind11::literals;

// jdate functions
namespace jdate
{
    double to_jdate_from_ktime(double k);
    double to_ktime_from_jdate(double j);
    double to_jdate_from_date_time(int m, int d, int y, double t);
    py::tuple get_month_day_year(double j);
    double get_time(double j);
}

// ephemeris functions
namespace ephemeris
{
    void build_ephemeris(py::dict py_info, py::list py_bodies);
    py::dict load_ephemeris(std::string file_name);
}

namespace astrodynamics
{
    py::tuple py_kepler(py::array_t<double> py_r0, py::array_t<double> py_v0, double t, double mu, std::optional<double> eps);
    py::tuple py_lambert(py::array_t<double> py_r0, py::array_t<double> py_r1, double t, double mu, std::optional<double> eps, std::optional<double> d, std::optional<py::array_t<double>> py_n);
    double splev(double x, py::object obj);
    double bisplev(double x, double y, py::object obj);
}

namespace interplanetary
{
    py::dict interplanetary(py::dict py_p, int mode, bool show_errors);
    py::tuple trajectory_correction(py::array_t<double> py_r, py::array_t<double> py_v, py::array_t<double> py_rt, py::array_t<double> py_vt, py::array_t<double> py_t, int bref, py::tuple py_body_info);
}

namespace kerbal_guidance_system
{
    void solve_atm_ivp(double t, double tout, py::array_t<double> py_y, py::dict py_p);
    void solve_vac_ivp(double t, double tout, py::object py_y, py::dict py_p);
    py::array_t<double> output_atm_ivp(double t, double tout, int steps, py::object py_y, py::dict py_p);
    py::array_t<double> output_vac_ivp(double t, double tout, int steps, py::object py_y, py::dict py_p);
    void solve_orbit_burn(double t, double tout, py::object py_y, py::dict py_p);
}
