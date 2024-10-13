#include <iostream>
#include <iomanip>
#include <Eigen/Dense>

#include "py_lib.h"

#include "Equation.h"
#include "Spl.h"
#include "Scalar.h"

namespace py_util
{
	static size_t array_size(py::object obj)
	{
		return obj.cast <py::array_t<double>>().size();
	}

	static double* array_ptr(py::object obj)
	{
		return obj.cast <py::array_t<double>>().mutable_data();
	}

	static void array_copy(py::array_t<double> src, py::array_t<double> dst)
	{
		dst.resize({ src.size() });
		std::copy(src.mutable_data(), src.mutable_data() + src.size(), dst.mutable_data());
	}

	static void array_copy(std::vector<double> src, py::array_t<double> dst)
	{
		dst.resize({ src.size() });
		std::copy(src.data(), src.data() + src.size(), dst.mutable_data());
	}

	static Eigen::Vector3d vector3d(py::object obj, int n)
	{
		return Eigen::Vector3d(obj.cast<py::array_t<double>>().mutable_data() + n);
	}

	static Spl spline(py::object obj)
	{
		py::tuple tup = obj;
		return Spl(array_ptr(tup[0]), array_ptr(tup[1]), array_size(tup[0]), tup[2].cast<size_t>());
	}
}

namespace kerbal_guidance_system
{
	struct AtmParams {
		Scalar scalar;
		Spl pressure;
		Spl density;
		Spl drag;
		Spl drag_mul;
		Spl thrust;
		double* angular_velocity;
		double a_limit;
		double azimuth;
		double pitch_time;
		double pitch_rate;
		double pitch_max;
		double mass_rate;

		AtmParams(py::dict py_params) :
			scalar(Scalar(
				py_params["scalars"]["time"].cast<double>(),
				py_params["scalars"]["distance"].cast<double>(),
				py_params["scalars"]["mass"].cast<double>())),
			pressure(py_util::spline(py_params["splines"]["pressure"])),
			density(py_util::spline(py_params["splines"]["density"])),
			drag(py_util::spline(py_params["splines"]["drag"])),
			drag_mul(py_util::spline(py_params["splines"]["drag_mul"])),
			thrust(Spl()),
			angular_velocity(py_params["settings"]["body_angular_velocity"].cast<py::array_t<double>>().mutable_data()),
			a_limit(py_params["settings"]["a_limit"].cast<double>()),
			azimuth(py_params["settings"]["azimuth"].cast<double>()),
			pitch_time(py_params["settings"]["pitch_time"].cast<double>()),
			pitch_rate(py_params["settings"]["pitch_rate"].cast<double>()),
			pitch_max(py_params["settings"]["pitch_max"].cast<double>()),
			mass_rate(0.0)
		{};
	};

	struct VacParams {
		Eigen::Vector3d li;
		Eigen::Vector3d dli;
		double tgi;
		double a_limit;
		bool coast;
		double thrust;
		double mass_rate;

		VacParams(py::dict py_params) :
			li(py_util::vector3d(py_params["settings"]["u"], 0)),
			dli(py_util::vector3d(py_params["settings"]["u"], 3)),
			tgi(py_params["settings"]["tg"].cast<double>()),
			a_limit(py_params["settings"]["a_limit"].cast<double>()),
			coast(false),
			thrust(0.0),
			mass_rate(0.0)
		{};
	};

	struct SimVacParams {
		VacParams vac_params;
		Scalar scalar;
		std::string integrator;
		double reltol;
		double abstol;
		double coast_time;
		double coast_duration;
		double tgf;

		SimVacParams(py::dict py_params) :
			vac_params(VacParams(py_params)),
			scalar(Scalar(
				py_params["scalars"]["time"].cast<double>(),
				py_params["scalars"]["distance"].cast<double>(),
				py_params["scalars"]["mass"].cast<double>())),
			integrator(py_params["settings"]["integrator"].cast<std::string>()),
			reltol(py_params["settings"]["reltol"].cast<double>()),
			abstol(py_params["settings"]["abstol"].cast<double>()),
			coast_time(py_params["settings"]["coast_time"].cast<double>()),
			coast_duration(py_params["settings"]["coast_duration"].cast<double>()),
			tgf(py_params["settings"]["tg"].cast<double>() + py_params["settings"]["u"].cast<py::array_t<double>>().at(6))
		{}
	};

	static void atm_state_derivatives(double t, double y[], double yp[], void* params)
	{
		// params
		AtmParams* atm_params = reinterpret_cast<AtmParams*>(params);

		// state
		Eigen::Map<Eigen::Vector3d> r(y);
		Eigen::Map<Eigen::Vector3d> v(y + 3);
		double& m = y[6];

		// state derivatives
		Eigen::Map<Eigen::Vector3d> dr(yp);
		Eigen::Map<Eigen::Vector3d> dv(yp + 3);
		double& dm = yp[6];

		double rnorm = r.norm();
		double vnorm = v.norm();
		double rdim_rnorm = atm_params->scalar.rdim(Scalar::Quantity::DISTANCE, rnorm);
		double rdim_vnorm = atm_params->scalar.rdim(Scalar::Quantity::VELOCITY, vnorm);

		Eigen::Map<Eigen::Vector3d> angular_velocity(atm_params->angular_velocity);

		Eigen::Vector3d attitude = r.normalized();
		if (t > atm_params->pitch_time)
		{
			double position_velocity_angle = acos(std::clamp(r.dot(v) / rnorm / vnorm, -1.0, 1.0));
			double pitch_over_angle = std::min(atm_params->pitch_rate * (t - atm_params->pitch_time), atm_params->pitch_max);
			double pitch = std::max(pitch_over_angle, position_velocity_angle);
			Eigen::Vector3d axis = Eigen::AngleAxisd(atm_params->azimuth, r.normalized()) * Eigen::Vector3d::UnitY().cross(r).cross(r).normalized();
			attitude = Eigen::AngleAxisd(pitch, axis) * attitude;
		}

		double pressure = exp(atm_params->pressure.eval(rdim_rnorm));
		double atmospheres = pressure / 101325.0;
		double density = exp(atm_params->density.eval(rdim_rnorm));
		double mach = rdim_vnorm / sqrt(1.4 * pressure / density);
		double drag_force = atm_params->drag.eval(mach) * atm_params->drag_mul.eval(density * rdim_vnorm) * density * rdim_vnorm * rdim_vnorm;
		double thrust = atm_params->thrust.eval(atmospheres);

		drag_force = atm_params->scalar.ndim(Scalar::Quantity::FORCE, drag_force);
		thrust = atm_params->scalar.ndim(Scalar::Quantity::FORCE, thrust);

		double throttle = std::min(1.0, m * atm_params->a_limit / thrust);

		dr = v;
		dv = attitude * throttle * thrust / m - r / pow(rnorm, 3.0) - v / vnorm * drag_force / m - 2.0 * angular_velocity.cross(v) - angular_velocity.cross(angular_velocity.cross(r));
		yp[6] = -throttle * atm_params->mass_rate;
	}

	static void simulate_atm_phase(double& t, py::array_t<double>& py_y, py::list& py_events, py::dict& py_params, std::vector<double>* output)
	{
		std::string integrator = py_params["settings"]["integrator"].cast<std::string>();
		double reltol = py_params["settings"]["reltol"].cast<double>();
		double abstol = py_params["settings"]["abstol"].cast<double>();

		AtmParams atm_params = AtmParams(py_params);

		bool altitude_reached = false;
		double switch_altitude = py_params["settings"]["switch_altitude"].cast<double>();

		double* y = py_y.mutable_data();
		Eigen::Map<Eigen::Vector3d> r(y);
		Eigen::Map<Eigen::Vector3d> v(y + 3);

		for (size_t i = 0; i < py_events.size(); i++)
		{
			py::tuple py_event = py_events[i];

			bool stage = py_event.attr("stage").cast<bool>();
			if (stage) y[6] = atm_params.scalar.ndim(Scalar::Quantity::MASS, py_event.attr("mf").cast<double>());

			atm_params.thrust = py_event.attr("spl_thrust").is_none() ? Spl() : py_util::spline(py_event.attr("spl_thrust"));
			atm_params.mass_rate = atm_params.scalar.ndim(Scalar::Quantity::MASS_RATE, py_event.attr("mdot").cast<double>());

			Equation eq = Equation(atm_state_derivatives, 7, t, y, integrator, reltol, abstol, &atm_params);

			double tf = atm_params.scalar.ndim(Scalar::Quantity::TIME, py_event.attr("tf").cast<double>());
			size_t steps = size_t(atm_params.scalar.rdim(Scalar::Quantity::TIME, tf - t)) + 1;
			double dt = (tf - t) / steps;

			for (size_t step = 0; step <= steps; step++)
			{
				if (tf < t) throw;
				eq.stepn(t + step * dt, tf);
				eq.get_y(0, 7, y);

				if (r.dot(v) < 0) throw std::runtime_error("Switch altitude not reached: negative altitude rate.");

				if (r.norm() > switch_altitude)
				{
					for (size_t iter = 0; iter < 10; iter++)
					{
						double f = r.norm() - switch_altitude;
						double df = r.dot(v) / r.norm();
						eq.step(eq.get_t() - f / df);
						eq.get_y(0, 7, y);
						if (abs(f) < 1e-8)
						{
							Eigen::Map<Eigen::Vector3d> angular_velocity(atm_params.angular_velocity);
							double angle = py_params["settings"]["initial_angle"].cast<double>() + eq.get_t() * angular_velocity.norm();
							Eigen::Vector3d axis = angular_velocity.normalized();
							Eigen::AngleAxisd rotation = Eigen::AngleAxisd(angle, axis);
							r = rotation * r;
							v = rotation * v + angular_velocity.cross(r);
							altitude_reached = true;
							break;
						}
					}
					if (!altitude_reached) throw std::runtime_error("Switch altitude not reached: refinement iterations exceeded.");
				}

				if (output)
				{
					output->push_back(i);
					output->push_back(eq.get_tot_iter());
					output->push_back(eq.get_rej_iter());
					output->push_back(eq.get_t());
					for (size_t i = 0; i < 7; i++)
					{
						output->push_back(eq.get_y(i));
					}
					for (size_t i = 3; i < 6; i++)
					{
						output->push_back(eq.get_yp(i));
					}
				}

				if (altitude_reached) break;
			}
			t = eq.get_t();
			if (altitude_reached) return;
		}
	}

	static void rdim_atm_output(std::vector<double>& output, py::array_t<double>& py_rdim_output, py::dict& py_params)
	{
		Scalar::Quantity scalar_quantities[] = {
			Scalar::Quantity::TIME,
			Scalar::Quantity::DISTANCE,
			Scalar::Quantity::DISTANCE,
			Scalar::Quantity::DISTANCE,
			Scalar::Quantity::VELOCITY,
			Scalar::Quantity::VELOCITY,
			Scalar::Quantity::VELOCITY,
			Scalar::Quantity::MASS,
			Scalar::Quantity::ACCELERATION,
			Scalar::Quantity::ACCELERATION,
			Scalar::Quantity::ACCELERATION };

		Scalar scalar = Scalar(
			py_params["scalars"]["time"].cast<double>(),
			py_params["scalars"]["distance"].cast<double>(),
			py_params["scalars"]["mass"].cast<double>());

		double* rdim_output = py_rdim_output.mutable_data();
		for (size_t i = 0; i < int(output.size() / 14); i++)
		{
			for (size_t j = 0; j < 3; j++) rdim_output[26 * i + j] = output[14 * i + j];

			for (size_t j = 0; j < 11; j++) rdim_output[26 * i + j + 3] = scalar.rdim(scalar_quantities[j], output[14 * i + j + 3]);

			Eigen::Map<Eigen::Vector3d> r(&output[14 * i + 4]);
			Eigen::Map<Eigen::Vector3d> v(&output[14 * i + 7]);
			double& m = output[14 * i + 10];
			Eigen::Map<Eigen::Vector3d> a(&output[14 * i + 11]);

			double rnorm = r.norm();
			double vnorm = v.norm();
			double rdim_rnorm = scalar.rdim(Scalar::Quantity::DISTANCE, rnorm);
			double rdim_vnorm = scalar.rdim(Scalar::Quantity::VELOCITY, vnorm);

			Eigen::Vector3d angular_velocity = Eigen::Vector3d(py_params["settings"]["body_angular_velocity"].cast<py::array_t<double>>().mutable_data());
			Eigen::Vector3d rdim_angular_velocity = scalar.rdim(Scalar::Quantity::RATE, 1.0) * angular_velocity;

			Spl spl_pressure = py_util::spline(py_params["splines"]["pressure"]);
			Spl spl_density = py_util::spline(py_params["splines"]["density"]);
			Spl spl_drag = py_util::spline(py_params["splines"]["drag"]);
			Spl spl_drag_mul = py_util::spline(py_params["splines"]["drag_mul"]);

			double pressure = exp(spl_pressure.eval(rdim_rnorm));
			double atmospheres = pressure / 101325.0;
			double density = exp(spl_density.eval(rdim_rnorm));
			double mach = rdim_vnorm / sqrt(1.4 * pressure / density);
			double drag_force = spl_drag.eval(mach) * spl_drag_mul.eval(density * rdim_vnorm) * density * rdim_vnorm * rdim_vnorm;
	
			Eigen::Vector3d drag_acceleration = -v / vnorm * drag_force / scalar.rdim(Scalar::Quantity::MASS, m);
			Eigen::Vector3d grav_acceleration = scalar.rdim(Scalar::Quantity::ACCELERATION, 1.0) * -r / pow(rnorm, 3.0);
			Eigen::Vector3d rot_acceleration = scalar.rdim(Scalar::Quantity::ACCELERATION, 1.0) * (- 2.0 * angular_velocity.cross(v) - angular_velocity.cross(angular_velocity.cross(r)));
			Eigen::Vector3d thrust_acceleration = scalar.rdim(Scalar::Quantity::ACCELERATION, 1.0) * a - drag_acceleration - grav_acceleration - rot_acceleration;
	
			for (size_t j = 0; j < 3; j++) rdim_output[26 * i + j + 14] = drag_acceleration(j);
			for (size_t j = 0; j < 3; j++) rdim_output[26 * i + j + 17] = grav_acceleration(j);
			for (size_t j = 0; j < 3; j++) rdim_output[26 * i + j + 20] = rot_acceleration(j);
			for (size_t j = 0; j < 3; j++) rdim_output[26 * i + j + 23] = thrust_acceleration(j);

			double t_output = rdim_output[26 * i + 3];
			Eigen::Map<Eigen::Vector3d> r_output(&rdim_output[26 * i + 4]);
			Eigen::Map<Eigen::Vector3d> v_output(&rdim_output[26 * i + 7]);
			Eigen::Map<Eigen::Vector3d> a_output(&rdim_output[26 * i + 11]);
			Eigen::Map<Eigen::Vector3d> d_output(&rdim_output[26 * i + 14]);
			Eigen::Map<Eigen::Vector3d> g_output(&rdim_output[26 * i + 17]);
			Eigen::Map<Eigen::Vector3d> o_output(&rdim_output[26 * i + 20]);
			Eigen::Map<Eigen::Vector3d> f_output(&rdim_output[26 * i + 23]);

			double angle = py_params["settings"]["initial_angle"].cast<double>() + t_output * rdim_angular_velocity.norm();
			Eigen::Vector3d axis = rdim_angular_velocity.normalized();
			Eigen::AngleAxisd rotation = Eigen::AngleAxisd(angle, axis);

			r_output = rotation * r_output;
			v_output = rotation * v_output + angular_velocity.cross(r_output);
			a_output = rotation * a_output;
			d_output = rotation * d_output;
			g_output = rotation * g_output;
			o_output = rotation * o_output;
			f_output = rotation * f_output;
		}
	}

	static void vac_state_derivatives(double t, double y[], double yp[], void* params)
	{
		// state
		Eigen::Map<Eigen::Vector3d> r(y);
		Eigen::Map<Eigen::Vector3d> v(y + 3);
		double& m = y[6];

		// state derivatives
		Eigen::Map<Eigen::Vector3d> dr(yp);
		Eigen::Map<Eigen::Vector3d> dv(yp + 3);
		double& dm = yp[6];

		// params
		VacParams* vac_params = reinterpret_cast<VacParams*>(params);

		// guidance and a limit
		double dt = t - vac_params->tgi;
		double thrust = vac_params->coast ? 0.0 : std::min(m * vac_params->a_limit, vac_params->thrust);
		double throttle = thrust / vac_params->thrust;
		Eigen::Vector3d l = dt > 0.0 ? vac_params->li * cos(dt) + vac_params->dli * sin(dt) : Eigen::Vector3d(v);

		// set derivatives
		dr = v;
		dv = -r / pow(r.norm(), 3.0) + thrust / m * l.normalized();
		dm = -throttle * vac_params->mass_rate;
	}

	static void vac_state_derivatives_guidance(double t, double y[], double yp[], void* params)
	{
		// state
		Eigen::Map<Eigen::Vector3d> r(y);
		Eigen::Map<Eigen::Vector3d> v(y + 3);
		double& m = y[6];
		
		// state derivatives
		Eigen::Map<Eigen::Vector3d> dr(yp);
		Eigen::Map<Eigen::Vector3d> dv(yp + 3);
		double& dm = yp[6];

		// state transition matrix
		Eigen::Map<Eigen::Matrix<double, 13, 13>> stm(y + 7);

		// state transition matrix derivatives
		Eigen::Map<Eigen::Matrix<double, 13, 13>> dstm(yp + 7);
		Eigen::Ref<Eigen::Matrix3d> dr_dv = dstm.block<3, 3>(0, 3);
		Eigen::Ref<Eigen::Matrix3d> dv_dr = dstm.block<3, 3>(3, 0);
		Eigen::Ref<Eigen::Vector3d> dv_dm = dstm.block<3, 1>(3, 6);
		Eigen::Ref<Eigen::Matrix<double, 1, 1>> dm_dm = dstm.block<1, 1>(6, 6);
		Eigen::Ref<Eigen::Matrix3d> dv_dl = dstm.block<3, 3>(3, 7);
		Eigen::Ref<Eigen::Matrix3d> dl_dld = dstm.block<3, 3>(7, 10);
		Eigen::Ref<Eigen::Matrix3d> dld_dl = dstm.block<3, 3>(10, 7);

		// params
		VacParams* vac_params = reinterpret_cast<VacParams*>(params);

		// guidance and a limit
		double dt = t - vac_params->tgi;
		double thrust = std::min(m * vac_params->a_limit, vac_params->thrust);
		double throttle = thrust / vac_params->thrust;
		Eigen::Vector3d l = vac_params->li * cos(dt) + vac_params->dli * sin(dt);

		// set derivatives
		dr = v;
		dv = -r / pow(r.norm(), 3.0) + throttle * l.normalized() * vac_params->thrust / m;
		dm = -throttle * vac_params->mass_rate;

		// set state transition matrix derivatives
		dstm.setZero();

		dr_dv.setIdentity();
		dv_dr = 3.0 * r * r.transpose() * pow(r.norm(), -5.0) - Eigen::Matrix3d::Identity() * pow(r.norm(), -3.0);
		dv_dm = l.normalized() * (throttle < 1.0 ? 0.0 : -vac_params->thrust / m / m);
		dm_dm << (throttle < 1.0 ? -vac_params->mass_rate * vac_params->a_limit / vac_params->thrust : 0.0);
		dv_dl = throttle * vac_params->thrust / m * (-l * l.transpose() + Eigen::Matrix3d::Identity() * l.squaredNorm()) * pow(l.norm(), -3.0);

		dl_dld.setIdentity();
		dld_dl.setIdentity();
		dld_dl = -dld_dl;

		dstm *= stm;
	}

	static void vac_state_derivatives_coast(double t, double y[], double yp[], void* params)
	{
		// state
		Eigen::Map<Eigen::Vector3d> r(y);
		Eigen::Map<Eigen::Vector3d> v(y + 3);
		double& m = y[6];

		// state derivatives
		Eigen::Map<Eigen::Vector3d> dr(yp);
		Eigen::Map<Eigen::Vector3d> dv(yp + 3);
		double& dm = yp[6];

		// state transition matrix
		Eigen::Map<Eigen::Matrix<double, 6, 6>> stm(y + 7);

		// state transition matrix derivatives
		Eigen::Map<Eigen::Matrix<double, 6, 6>> dstm(yp + 7);
		Eigen::Ref<Eigen::Matrix3d> dr_dv = dstm.block<3, 3>(0, 3);
		Eigen::Ref<Eigen::Matrix3d> dv_dr = dstm.block<3, 3>(3, 0);

		// set derivatives
		dr = v;
		dv = -r / pow(r.norm(), 3.0);
		dm = 0.0;

		// set state transition matrix derivatives
		dstm.setZero();
		dr_dv.setIdentity();
		dv_dr = 3.0 * r * r.transpose() * pow(r.norm(), -5.0) - Eigen::Matrix3d::Identity() * pow(r.norm(), -3.0);

		dstm *= stm;
	}

	static void simulate_vac_phase(double& t, double tout, double* y, bool coast, py::list& py_events, SimVacParams sim_vac_params)
	{
		sim_vac_params.vac_params.coast = coast;
		for (size_t i = 0; i < py_events.size(); i++)
		{
			py::tuple py_event = py_events[i];

			double tf = sim_vac_params.scalar.ndim(Scalar::Quantity::TIME, py_event.attr("tf").cast<double>());
			if (tf < t) continue;

			bool stage = py_event.attr("stage").cast<bool>();
			if (stage) y[6] = sim_vac_params.scalar.ndim(Scalar::Quantity::MASS, py_event.attr("mf").cast<double>());

			sim_vac_params.vac_params.thrust = sim_vac_params.scalar.ndim(Scalar::Quantity::FORCE, py_event.attr("thrust_vac").cast<double>());
			sim_vac_params.vac_params.mass_rate = sim_vac_params.scalar.ndim(Scalar::Quantity::MASS_RATE, py_event.attr("mdot").cast<double>());

			Equation eq = Equation(vac_state_derivatives, 7, t, y, sim_vac_params.integrator, sim_vac_params.reltol, sim_vac_params.abstol, &sim_vac_params.vac_params);
			eq.stepn(std::min(tf, tout), tf);
			t = eq.get_t();
			eq.get_y(0, 7, y);
			if (tf > tout) return;
		}
	}

	static void simulate_vac_phase_to_velocity(double& t, double vout, double* y, py::list& py_events, SimVacParams sim_vac_params)
	{
		Eigen::Map<Eigen::Vector3d> r(y);
		Eigen::Map<Eigen::Vector3d> v(y + 3);
		Eigen::Vector3d a;
		for (size_t i = 0; i < py_events.size(); i++)
		{
			py::tuple py_event = py_events[i];

			double tf = sim_vac_params.scalar.ndim(Scalar::Quantity::TIME, py_event.attr("tf").cast<double>());
			if (tf < t) continue;

			bool stage = py_event.attr("stage").cast<bool>();
			if (stage) y[6] = sim_vac_params.scalar.ndim(Scalar::Quantity::MASS, py_event.attr("mf").cast<double>());

			sim_vac_params.vac_params.thrust = sim_vac_params.scalar.ndim(Scalar::Quantity::FORCE, py_event.attr("thrust_vac").cast<double>());
			sim_vac_params.vac_params.mass_rate = sim_vac_params.scalar.ndim(Scalar::Quantity::MASS_RATE, py_event.attr("mdot").cast<double>());

			Equation eq = Equation(vac_state_derivatives, 7, t, y, sim_vac_params.integrator, sim_vac_params.reltol, sim_vac_params.abstol, &sim_vac_params.vac_params);

			size_t steps = size_t(sim_vac_params.scalar.rdim(Scalar::Quantity::TIME, tf - t)) + 1;
			double dt = (tf - t) / steps;
			for (size_t step = 0; step <= steps; step++)
			{
				eq.stepn(t + step * dt, tf);
				eq.get_y(0, 7, y);

				if (v.norm() > vout)
				{
					for (size_t iter = 0; iter < 10; iter++)
					{
						eq.get_yp(3, 6, a.data());
						double f = v.norm() - vout;
						double df = v.dot(a) / v.norm();
						eq.step(eq.get_t() - f / df);
						eq.get_y(0, 7, y);
						if (abs(f) < 1e-8)
						{
							t = eq.get_t();
							return;
						}
					}
					throw std::runtime_error("Final velocity not reached: refinement iterations exceeded.");
				}
			}
		}
	}

	static void simulate_vac_phase_output(double& t, double tout, double* y, bool coast, py::list& py_events, SimVacParams sim_vac_params, std::vector<double>& output)
	{
		sim_vac_params.vac_params.coast = coast;
		for (size_t i = 0; i < py_events.size(); i++)
		{
			py::tuple py_event = py_events[i];

			double tf = sim_vac_params.scalar.ndim(Scalar::Quantity::TIME, py_event.attr("tf").cast<double>());
			if (tf < t) continue;

			bool stage = py_event.attr("stage").cast<bool>();
			if (stage) y[6] = sim_vac_params.scalar.ndim(Scalar::Quantity::MASS, py_event.attr("mf").cast<double>());

			sim_vac_params.vac_params.thrust = sim_vac_params.scalar.ndim(Scalar::Quantity::FORCE, py_event.attr("thrust_vac").cast<double>());
			sim_vac_params.vac_params.mass_rate = sim_vac_params.scalar.ndim(Scalar::Quantity::MASS_RATE, py_event.attr("mdot").cast<double>());

			Equation eq = Equation(vac_state_derivatives, 7, t, y, sim_vac_params.integrator, sim_vac_params.reltol, sim_vac_params.abstol, &sim_vac_params.vac_params);

			size_t steps = size_t(sim_vac_params.scalar.rdim(Scalar::Quantity::TIME, tf - t)) + 1;
			double dt = (tf - t) / steps;
			for (size_t step = 0; step <= steps; step++)
			{
				eq.stepn(std::min(t + step * dt, tout), tf);
				eq.get_y(0, 7, y);

				output.push_back(i);
				output.push_back(eq.get_tot_iter());
				output.push_back(eq.get_rej_iter());
				output.push_back(eq.get_t());
				for (size_t i = 0; i < 7; i++) output.push_back(eq.get_y(i));
				for (size_t i = 3; i < 6; i++) output.push_back(eq.get_yp(i));

				if (t + step * dt > tout)
				{
					t = eq.get_t();
					return;
				}
			}
		}
	}

	static void simulate_vac_phase_stm(double& t, double tout, double* y, bool coast, py::list& py_events, SimVacParams sim_vac_params)
	{
		sim_vac_params.vac_params.coast = coast;
		if (coast)
		{
			Eigen::Map<Eigen::Matrix<double, 6, 6>> stm(y + 7);
			stm.setIdentity();
		}
		else
		{
			Eigen::Map<Eigen::Matrix<double, 13, 13>> stm(y + 7);
			stm.setIdentity();
		}
		for (size_t i = 0; i < py_events.size(); i++)
		{
			py::tuple py_event = py_events[i];

			double tf = sim_vac_params.scalar.ndim(Scalar::Quantity::TIME, py_event.attr("tf").cast<double>());
			if (tf < t) continue;

			bool stage = py_event.attr("stage").cast<bool>();
			if (stage) y[6] = sim_vac_params.scalar.ndim(Scalar::Quantity::MASS, py_event.attr("mf").cast<double>());

			sim_vac_params.vac_params.thrust = sim_vac_params.scalar.ndim(Scalar::Quantity::FORCE, py_event.attr("thrust_vac").cast<double>());
			sim_vac_params.vac_params.mass_rate = sim_vac_params.scalar.ndim(Scalar::Quantity::MASS_RATE, py_event.attr("mdot").cast<double>());

			Equation eq;
			if (coast)
			{
				eq = Equation(vac_state_derivatives_coast, 43, t, y, sim_vac_params.integrator, sim_vac_params.reltol, sim_vac_params.abstol, &sim_vac_params.vac_params);
			}
			else
			{
				eq = Equation(vac_state_derivatives_guidance, 176, t, y, sim_vac_params.integrator, sim_vac_params.reltol, sim_vac_params.abstol, &sim_vac_params.vac_params);
			}
			eq.stepn(std::min(tf, tout), tf);
			t = eq.get_t();
			eq.get_y(0, 7, y);
			if (tf > tout) return;
		}
	}

	static void rdim_vac_output(std::vector<double>& output, py::array_t<double>& py_rdim_output, py::dict& py_params)
	{
		Scalar::Quantity scalar_quantities[] = {
			Scalar::Quantity::TIME,
			Scalar::Quantity::DISTANCE,
			Scalar::Quantity::DISTANCE,
			Scalar::Quantity::DISTANCE,
			Scalar::Quantity::VELOCITY,
			Scalar::Quantity::VELOCITY,
			Scalar::Quantity::VELOCITY,
			Scalar::Quantity::MASS,
			Scalar::Quantity::ACCELERATION,
			Scalar::Quantity::ACCELERATION,
			Scalar::Quantity::ACCELERATION };

		Scalar scalar = Scalar(
			py_params["scalars"]["time"].cast<double>(),
			py_params["scalars"]["distance"].cast<double>(),
			py_params["scalars"]["mass"].cast<double>());

		double* rdim_output = py_rdim_output.mutable_data();
		for (size_t i = 0; i < int(output.size() / 14); i++)
		{
			for (size_t j = 0; j < 3; j++) rdim_output[26 * i + j] = output[14 * i + j];

			for (size_t j = 0; j < 11; j++) rdim_output[26 * i + j + 3] = scalar.rdim(scalar_quantities[j], output[14 * i + j + 3]);

			Eigen::Map<Eigen::Vector3d> r(&output[14 * i + 4]);
			Eigen::Map<Eigen::Vector3d> v(&output[14 * i + 7]);
			Eigen::Map<Eigen::Vector3d> a(&output[14 * i + 11]);

			Eigen::Vector3d drag_acceleration = Eigen::Vector3d::Zero();
			Eigen::Vector3d grav_acceleration = scalar.rdim(Scalar::Quantity::ACCELERATION, 1.0) * -r / pow(r.norm(), 3.0);
			Eigen::Vector3d rot_acceleration = Eigen::Vector3d::Zero();
			Eigen::Vector3d thrust_acceleration = scalar.rdim(Scalar::Quantity::ACCELERATION, 1.0) * a - grav_acceleration;

			for (size_t j = 0; j < 3; j++) rdim_output[26 * i + j + 14] = drag_acceleration(j);
			for (size_t j = 0; j < 3; j++) rdim_output[26 * i + j + 17] = grav_acceleration(j);
			for (size_t j = 0; j < 3; j++) rdim_output[26 * i + j + 20] = rot_acceleration(j);
			for (size_t j = 0; j < 3; j++) rdim_output[26 * i + j + 23] = thrust_acceleration(j);
		}
	}

	py::tuple py_simulate_atm_phase(double t, py::array_t<double> py_yi, py::list py_events, py::dict py_params)
	{
		py::array_t<double> py_y;
		py_util::array_copy(py_yi, py_y);
		simulate_atm_phase(t, py_y, py_events, py_params, nullptr);
		return py::make_tuple(t, py_y);
	}

	py::array_t<double> py_output_atm_phase(double t, py::array_t<double> py_yi, py::list py_events, py::dict py_params)
	{
		std::vector<double> output;
		output.reserve(1024);
		py::array_t<double> py_y;
		py_util::array_copy(py_yi, py_y);
		simulate_atm_phase(t, py_y, py_events, py_params, &output);
		py::array_t<double> py_rdim_output({ int(output.size() / 14), 26 });
		rdim_atm_output(output, py_rdim_output, py_params);
		return py_rdim_output;
	}

	py::tuple py_simulate_vac_phase(double t, py::array_t<double> py_yi, py::list py_events, py::dict py_params)
	{
		SimVacParams sim_vac_params = SimVacParams(py_params);

		py::array_t<double> py_y;
		py_util::array_copy(py_yi, py_y);
		double* y = py_y.mutable_data();

		simulate_vac_phase(t, sim_vac_params.coast_time, y, false, py_events, sim_vac_params);
		simulate_vac_phase(t, sim_vac_params.coast_time + sim_vac_params.coast_duration, y, true, py_events, sim_vac_params);
		simulate_vac_phase(t, sim_vac_params.tgf, y, false, py_events, sim_vac_params);

		return py::make_tuple(t, py_y);
	}

	py::tuple py_simulate_vac_phase_to_velocity(double t, py::array_t<double> py_yi, py::list py_events, py::dict py_params)
	{
		SimVacParams sim_vac_params = SimVacParams(py_params);

		double vout = py_params["settings"]["vout"].cast<double>();
		py::array_t<double> py_y;
		py_util::array_copy(py_yi, py_y);
		double* y = py_y.mutable_data();

		simulate_vac_phase_to_velocity(t, vout, y, py_events, sim_vac_params);

		return py::make_tuple(t, py_y);
	}

	py::array_t<double> py_output_vac_phase(double t, py::array_t<double> py_yi, py::list py_events, py::dict py_params)
	{
		SimVacParams sim_vac_params = SimVacParams(py_params);
		std::vector<double> output;
		output.reserve(1024);
		py::array_t<double> py_y;
		py_util::array_copy(py_yi, py_y);
		double* y = py_y.mutable_data();
		simulate_vac_phase_output(t, sim_vac_params.coast_time, y, false, py_events, sim_vac_params, output);
		simulate_vac_phase_output(t, sim_vac_params.coast_time + sim_vac_params.coast_duration, y, true, py_events, sim_vac_params, output);
		simulate_vac_phase_output(t, sim_vac_params.tgf, y, false, py_events, sim_vac_params, output);
		py::array_t<double> py_rdim_output({ int(output.size() / 14), 26 });
		rdim_vac_output(output, py_rdim_output, py_params);
		return py_rdim_output;
	}

	py::array_t<double> py_constraint_residuals(double t, py::array_t<double> py_yi, py::list py_events, py::dict py_params)
	{
		return py_yi;
	}

	py::array_t<double> py_constraint_jacobian(double t, py::array_t<double> py_yi, py::list py_events, py::dict py_params)
	{
		// calculate dy/dct
		// calculate dy/dy0 over coast
		// calculate dy/dcd

		// calculate full stm over guided phase

		py::array_t<double> py_y;
		py_y.resize({ 176 });
		std::copy(py_yi.mutable_data(), py_yi.mutable_data() + py_yi.size(), py_y.mutable_data());
		std::cout << (t > 100000000000.0) << '\n';
		return py_y;
		// simulate_vac_phase_stm_vguidance(t, py_y, py_events, py_params);
		return py_y;
	}
}