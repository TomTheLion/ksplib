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

template<typename T>
static void py_array_copy(py::array_t<T> src, py::array_t<T> dst)
{
	dst.resize(src.size());
	std::copy(src.mutable_data(), src.mutable_data() + src.size(), dst.mutable_data());
}

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
	struct Scalars
	{
		double time;
		double mass;
		double distance;
		double velocity;
		double acceleration;
	};

	struct Engine
	{
		Spl thrust;
		double vac_thrust;
		double mass_rate;
	};

	struct FlightEvent
	{
		bool stage;
		double tout;
		double mi;
		double mf;
		double throttle;
		Engine engine;
	};

	struct AtmParams
	{
		Scalars scalars;
		Spl pressure;
		Spl density;
		Spl drag;
		Spl drag_mul;
		Engine engine;
		double a_limit;
		double azimuth;
		double pitch_time;
		double pitch_rate;
		double pitch_max;
		double* body_angular_velocity;
	};

	struct VacParams
	{
		Engine engine;
		double a_limit;
		double tg;
		double* u;
	};

	static std::vector<FlightEvent> create_events(py::list py_events)
	{
		size_t n = py_events.size();
		std::vector<FlightEvent> flight_events(n);

		for (size_t i = 0; i < n; i++)
		{
			flight_events[i] = FlightEvent{
				py_events[i].attr("stage").cast<bool>(),
				py_events[i].attr("tout").cast<double>(),
				py_events[i].attr("mi").cast<double>(),
				py_events[i].attr("mf").cast<double>(),
				py_events[i].attr("throttle").cast<double>(),
				Engine{
					py_events[i].attr("spl_thrust").is_none() ? Spl() : py_spl(py_events[i].attr("spl_thrust")),
					py_events[i].attr("vac_thrust").cast<double>(),
					py_events[i].attr("mdot").cast<double>()
				}				
			};
		}

		return flight_events;
	}

	static AtmParams create_atm_params(py::dict py_params)
	{
		return AtmParams{
			Scalars{
				py_params["scalars"]["time"].cast<double>(),
				py_params["scalars"]["mass"].cast<double>(),
				py_params["scalars"]["distance"].cast<double>(),
				py_params["scalars"]["velocity"].cast<double>(),
				py_params["scalars"]["acceleration"].cast<double>(),
			},
			py_spl(py_params["splines"]["pressure"]),
			py_spl(py_params["splines"]["density"]),
			py_spl(py_params["splines"]["drag"]),
			py_spl(py_params["splines"]["drag_mul"]),
			Engine(),
			py_params["settings"]["a_limit"].cast<double>(),
			py_params["settings"]["azimuth"].cast<double>(),
			py_params["settings"]["pitch_time"].cast<double>(),
			py_params["settings"]["pitch_rate"].cast<double>(),
			py_params["settings"]["pitch_max"].cast<double>(),
			py_params["settings"]["body_angular_velocity"].cast<py::array_t<double>>().mutable_data()
		};
	}

	static VacParams create_vac_params(py::dict py_params)
	{
		return VacParams{
			Engine(),
			py_params["settings"]["a_limit"].cast<double>(),
			py_params["settings"]["tg"].cast<double>(),
			py_params["settings"]["u"].cast<py::array_t<double>>().mutable_data()
		};
	}

	static void atm_state(double t, double y[], double yp[], void* params)
	{
		AtmParams* atm_params = reinterpret_cast<AtmParams*>(params);
		Eigen::Map<Eigen::Vector3d> r(y);
		Eigen::Map<Eigen::Vector3d> v(y + 3);
		Eigen::Map<Eigen::Vector3d> dr(yp);
		Eigen::Map<Eigen::Vector3d> dv(yp + 3);
		Eigen::Map<Eigen::Vector3d> body_angular_velocity(atm_params->body_angular_velocity);
		double m = y[6];

		double rnorm = r.norm();
		double vnorm = v.norm();

		Eigen::Vector3d attitude = r.normalized();
		if (t > atm_params->pitch_time)
		{
			double position_velocity_angle = acos(std::clamp(r.dot(v) / rnorm / vnorm, -1.0, 1.0));
			double pitch_over_angle = std::min(atm_params->pitch_rate * (t - atm_params->pitch_time), atm_params->pitch_max);
			double pitch = std::max(pitch_over_angle, position_velocity_angle);
			Eigen::Vector3d axis = Eigen::AngleAxisd(atm_params->azimuth, r.normalized()) * Eigen::Vector3d::UnitY().cross(r).cross(r).normalized();
			attitude = Eigen::AngleAxisd(pitch, axis) * attitude;
		}

		double pressure = atm_params->pressure.eval(rnorm);
		double atmospheres = pressure / 101325.0;
		double density = atm_params->density.eval(rnorm);
		double mach = vnorm / sqrt(1.4 * pressure / density);
		double drag = atm_params->drag.eval(mach) * atm_params->drag_mul.eval(density * vnorm) * density * vnorm * vnorm;

		double thrust = atm_params->engine.thrust.eval(atmospheres);
		double throttle = atm_params->a_limit ? std::min(1.0, atm_params->a_limit / (thrust / m)) : 1.0;

		Eigen::Vector3d a_thrust = attitude * throttle * thrust / m;
		Eigen::Vector3d a_gravity = -r / pow(rnorm, 3);
		Eigen::Vector3d a_drag = -v / vnorm * drag / m;
		Eigen::Vector3d a_rotation = -2.0 * body_angular_velocity.cross(v) - body_angular_velocity.cross(body_angular_velocity.cross(r));

		dr = v;
		dv = a_thrust + a_gravity + a_drag + a_rotation;
		yp[6] = -throttle * atm_params->engine.mass_rate;
	}

	static void simulate_atm_phase(double& t, double* y, std::vector<FlightEvent>& flight_events, py::dict& py_params, std::vector<double>* output)
	{
		AtmParams atm_params = create_atm_params(py_params);

		double pitch_time = py_params["pitch_time"].cast<double>();
		double switch_altitude = py_params["switch_altitude"].cast<double>();

		std::string integrator = py_params["atm_integrator"].cast<std::string>();
		double reltol = py_params["atm_reltol"].cast<double>();
		double abstol = py_params["atm_abstol"].cast<double>();

		bool last = false;
		double altitude = 0.0;
		double altitude_old = 0.0;

		for (auto& flight_event : flight_events)
		{
			if (flight_event.stage) y[6] = flight_event.mf;
			atm_params.engine = flight_event.engine;
			Equation eq = Equation(atm_state, t, 7, y, integrator, reltol, abstol, &atm_params);
			double dt = 0.01 * (flight_event.tout - t);

			for (size_t step = 0; step <= 100; step++)
			{
				double tout = t + step * dt;

				if (eq.get_t() > pitch_time)
				{
					eq.step(tout);
				}
				else if (tout < pitch_time)
				{
					eq.stepn(tout, pitch_time);
				}
				else
				{
					eq.stepn(pitch_time, pitch_time);
					eq.step(tout);
				}

				altitude_old = altitude;
				altitude = std::sqrt(eq.get_y(0) * eq.get_y(0) + eq.get_y(1) * eq.get_y(1) + eq.get_y(2) * eq.get_y(2));

				if (altitude < altitude_old)
				{
					throw std::runtime_error("Switch altitude not reached: negative altitude rate.");
				}

				if (altitude > switch_altitude)
				{
					for (size_t iter = 0; iter < 10; iter++)
					{
						double f = altitude - switch_altitude;
						double df = (altitude - altitude_old) / dt;
						dt = -f / df;
						tout = eq.get_t() + dt;
						eq.step(tout);
						altitude_old = altitude;
						altitude = std::sqrt(eq.get_y(0) * eq.get_y(0) + eq.get_y(1) * eq.get_y(1) + eq.get_y(2) * eq.get_y(2));
						if (abs(dt) < 1e-8)
						{
							last = true;
							break;
						}
					}
					if (!last) throw std::runtime_error("Switch altitude not reached: refinement iterations exceeded.");
				}

				if (output)
				{
					output->push_back(t);
					for (size_t i = 0; i < 7; i++)
					{
						output->push_back(eq.get_y(i));
					}
					for (size_t i = 3; i < 6; i++)
					{
						output->push_back(eq.get_yp(i));
					}
				}
				t = eq.get_t();
				eq.get_y(0, 7, y);
				if (last) return;
			}
		}
	}

	static py::tuple py_simulate_atm_phase(double t, py::array_t<double> py_yi, py::list py_events, py::dict py_params)
	{
		// we can calculate flight events twice, it would only be for output which really isn't that important
		// we should really be focusing on increasing the speed of the vac phase, we should probably be solving the root outselves
		// c++

		py::array_t<double> py_y;
		py_array_copy(py_yi, py_y);

		std::vector<FlightEvent> flight_events = create_events(py_events);

		simulate_atm_phase(t, py_y.mutable_data(), flight_events, py_params, nullptr);

		return py::make_tuple(t, py_y);
	}
}