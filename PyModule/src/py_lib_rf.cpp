#include <iostream>
#include <iomanip>

#include <Eigen/Dense>

#include "py_lib.h"
#include "Spl.h"
#include "Equation.h"

// output: do conversions at end
// number of steps, think about that (could be standardized)
// converting to and from do in outside function
// event do at begining

static Spl py_spl(py::object obj)
{
	py::tuple tup = obj;
	return Spl(
		tup[0].cast<py::array_t<double>>().mutable_data(),
		tup[1].cast<py::array_t<double>>().mutable_data(),
		static_cast<int>(tup[0].cast<py::array_t<double>>().size()),
		tup[2].cast<int>());
}

namespace kerbal_guidance_system
{
	struct AtmSplines
	{
		Spl pressure;
		Spl density;
		Spl drag;
		Spl drag_mul;
	};

	struct Engine
	{
		Spl thrust;
		double vac_thrust;
		double mass_rate;
	};

	struct AtmParams
	{
		AtmSplines splines;
		double a_limit;
		double azimuth;
		double switch_altitude;
		double mass_rate;
		double pitch_time;
		double pitch_rate;
		double pitch_duration;
		double initial_angle;
		Eigen::Vector3d angular_velocity;
		std::string atm_integrator;
		double atm_reltol;
		double atm_abstol;

	};

	struct AtmParams2
	{
		AtmSplines splines;
		Engine engine;
		double a_limit;
		double azimuth;
		double switch_altitude;
		double initial_angle;
		Eigen::Vector3d angular_velocity;
		std::string atm_integrator;
		double atm_reltol;
		double atm_abstol;

	};

	struct VacParams
	{
		double a_limit;
		double thrust;
		double mass_rate;
		double tg;
		double* u;
		std::string vac_integrator;
		double vac_reltol;
		double vac_abstol;
	};

	struct Event
	{
		bool stage;
		double tout;
		double mi; 
		double mf;
		double throttle;
		Engine engine;
	};

	static AtmParams create_atm_params(py::dict py_params)
	{

	}

	static VacParams create_vac_params(py::dict py_params)
	{
		VacParams vac_params{
			UnitScalars{
				py_params["unit_scalars"]["time"].cast<double>(),
				py_params["unit_scalars"]["distance"].cast<double>(),
				py_params["unit_scalars"]["mass"].cast<double>()},
			py_params["settings"]["a_limit"].cast<double>(),
			0.0,
			0.0,
			py_params["settings"]["tg"].cast<double>(),
			py_params["settings"]["u"].cast<py::array_t<double>>().mutable_data(),
			py_params["settings"]["vac_integrator"].cast<std::string>(),
			py_params["settings"]["vac_reltol"].cast<double>(),
			py_params["settings"]["vac_abstol"].cast<double>(),
		};
	}

	static std::vector<Event> create_events(py::list py_events)
	{
		size_t n = py_events.size();
		std::vector<Event> events(n);

		for (size_t i = 0; i < n; i++)
		{
			events[i] = Event{
				py_events[i].attr("stage").cast<bool>(),
				py_events[i].attr("tout").cast<double>(),
				py_events[i].attr("mi").cast<double>(),
				py_events[i].attr("mf").cast<double>(),
				py_events[i].attr("mdot").cast<double>(),
				py_events[i].attr("throttle").cast<double>(),
				py_events[i].attr("vac_thrust").cast<double>(),
				py_events[i].attr("spl_thrust").is_none() ? Spl() : py_spl(py_events[i].attr("spl_thrust"))
			};
		}

		return events;
	}

	py::tuple simulate_atm_phase(double t, py::array_t<double> py_yi, py::list py_events, py::dict py_p)
	{
		py::array_t<double> py_y;
		py_y.resize({ py_yi.size() });
		std::cout << py_yi.size() << '\n';
		std::cout << py_yi.mutable_unchecked().nbytes() << '\n';
		std::memcpy(py_y.mutable_data(), py_yi.mutable_data(), py_yi.mutable_unchecked().nbytes());

		double* y = py_y.mutable_data();

		y[0] = py_events[0].cast<double>();
		y[1] = py_p["a"].cast<double>();

		return py::make_tuple(t, py_y);
	}
}