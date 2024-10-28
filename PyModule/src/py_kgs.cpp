#include <iostream>
#include <iomanip>

#include "py_lib.h"

#include "Equation.h"
#include "Scalar.h"


namespace kerbal_guidance_system
{
	// structs
	struct AtmParams {
		Scalar scalar;
		py_util::Spl pressure;
		py_util::Spl density;
		py_util::Spl drag;
		py_util::Spl drag_mul;
		py_util::Spl thrust;
		Eigen::Vector3d angular_velocity;
		double a_limit;
		double azimuth;
		double pitch_time;
		double pitch_rate;
		double pitch_max;
		double thrust_vac;
		double mass_rate;

		AtmParams(double pitch_rate, double azimuth, py::dict& py_params) :
			scalar(Scalar(
				py_params["scalars"]["time"].cast<double>(),
				py_params["scalars"]["distance"].cast<double>(),
				py_params["scalars"]["mass"].cast<double>())),
			pressure(py_util::spline(py_params["splines"]["pressure"])),
			density(py_util::spline(py_params["splines"]["density"])),
			drag(py_util::spline(py_params["splines"]["drag"])),
			drag_mul(py_util::spline(py_params["splines"]["drag_mul"])),
			thrust(py_util::Spl()),
			angular_velocity(py_util::vector3d(py_params["settings"]["angular_velocity"])),
			a_limit(py_params["settings"]["a_limit"].cast<double>()),
			azimuth(azimuth),
			pitch_time(py_params["settings"]["pitch_time"].cast<double>()),
			pitch_rate(pitch_rate),
			pitch_max(pitch_rate * py_params["settings"]["pitch_duration"].cast<double>()),
			thrust_vac(0.0),
			mass_rate(0.0)
		{};
	};

	struct VacParams {
		Eigen::Vector3d li;
		Eigen::Vector3d ldi;
		bool const_accel;
		double tgi;
		double a_limit;
		double thrust;
		double mass_rate;

		VacParams(py::array_t<double>& py_x, py::dict& py_params) :
			li(py_util::vector3d(py_x)),
			ldi(py_util::vector3d(py_x, 3)),
			const_accel(false),
			tgi(py_x.at(7) + py_x.at(8)),
			a_limit(py_params["settings"]["a_limit"].cast<double>()),
			thrust(0.0),
			mass_rate(0.0)
		{};

		VacParams(py::array_t<double>& py_x, double tgi, py::dict& py_params) :
			li(py_util::vector3d(py_x)),
			ldi(py_util::vector3d(py_x, 3)),
			const_accel(false),
			tgi(tgi),
			a_limit(py_params["settings"]["a_limit"].cast<double>()),
			thrust(0.0),
			mass_rate(0.0)
		{};
	};

	struct IntegratorParams
	{
		std::string method;
		double reltol;
		double abstol;

		IntegratorParams(py::dict& py_params) :
			method(py_params["integrator"]["method"].cast<std::string>()),
			reltol(py_params["integrator"]["reltol"].cast<double>()),
			abstol(py_params["integrator"]["abstol"].cast<double>())
		{}
	};

	struct VehicleEvents {
		py::list py_events;
		size_t i;
		Scalar scalar;

		VehicleEvents(py::list& py_events, py::dict& py_params) :
			py_events(py_events), i(0),
			scalar(
				py_params["scalars"]["time"].cast<double>(),
				py_params["scalars"]["distance"].cast<double>(),
				py_params["scalars"]["mass"].cast<double>())
		{};

		void next()
		{
			if (i < py_events.size()) i++;
		}

		bool is_valid() const
		{
			return i < py_events.size();
		}

		int get_current() const
		{
			return i;
		}

		bool get_stage() const
		{
			return py_events[i].cast<py::tuple>()[0].cast<bool>();
		}

		bool get_const_accel() const
		{
			return py_events[i].cast<py::tuple>()[1].cast<bool>();
		}

		double get_tf() const
		{
			return scalar.ndim(Scalar::Quantity::TIME, py_events[i].cast<py::tuple>()[2].cast<double>());
		}

		double get_mi() const
		{
			return scalar.ndim(Scalar::Quantity::MASS, py_events[i].cast<py::tuple>()[3].cast<double>());
		}

		double get_mf() const
		{
			return scalar.ndim(Scalar::Quantity::MASS, py_events[i].cast<py::tuple>()[4].cast<double>());
		}

		double get_mdot() const
		{
			return scalar.ndim(Scalar::Quantity::MASS_RATE, py_events[i].cast<py::tuple>()[5].cast<double>());
		}

		double get_thrust_vac() const
		{
			return scalar.ndim(Scalar::Quantity::FORCE, py_events[i].cast<py::tuple>()[6].cast<double>());
		}

		py_util::Spl get_spl_thrust() const
		{
			return py_events[i].cast<py::tuple>()[7].is_none() ? py_util::Spl() : py_util::spline(py_events[i].cast<py::tuple>()[7]);
		}
	};

	// atmospheric functions
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

		// guidance
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

		// drag, thrust, and throttle
		double rdim_rnorm = atm_params->scalar.rdim(Scalar::Quantity::DISTANCE, rnorm);
		double rdim_vnorm = atm_params->scalar.rdim(Scalar::Quantity::VELOCITY, vnorm);

		double pressure = exp(atm_params->pressure.eval(rdim_rnorm));
		double atmospheres = pressure / 101325.0;
		double density = exp(atm_params->density.eval(rdim_rnorm));
		double mach = rdim_vnorm / sqrt(1.4 * pressure / density);
		double drag_force = atm_params->drag.eval(mach) * atm_params->drag_mul.eval(density * rdim_vnorm) * density * rdim_vnorm * rdim_vnorm;
		double thrust = atm_params->thrust.eval(atmospheres);

		drag_force = atm_params->scalar.ndim(Scalar::Quantity::FORCE, drag_force);
		thrust = atm_params->scalar.ndim(Scalar::Quantity::FORCE, thrust);
		double throttle = thrust > 0 ? std::min(1.0, m * atm_params->a_limit / atm_params->thrust_vac) : 0.0;

		// set derivatives
		dr = v;
		dv = attitude * throttle * thrust / m - r / pow(rnorm, 3.0) - v / vnorm * drag_force / m - 2.0 * atm_params->angular_velocity.cross(v) - atm_params->angular_velocity.cross(atm_params->angular_velocity.cross(r));
		yp[6] = -throttle * atm_params->mass_rate;
	}

	static void simulate_atm_phase(double& t, double* y, double switch_altitude, VehicleEvents& vehicle_events, AtmParams& atm_params, IntegratorParams& integrator_params, std::vector<double>* output)
	{
		Eigen::Map<Eigen::Vector3d> r(y);
		Eigen::Map<Eigen::Vector3d> v(y + 3);

		while (vehicle_events.is_valid())
		{
			if (vehicle_events.get_stage()) y[6] = vehicle_events.get_mf();
			atm_params.thrust = vehicle_events.get_spl_thrust();
			atm_params.mass_rate = vehicle_events.get_mdot();
			atm_params.thrust_vac = vehicle_events.get_thrust_vac();

			Equation eq = Equation(atm_state_derivatives, 7, t, y, integrator_params.method, integrator_params.reltol, integrator_params.abstol, &atm_params);

			auto output_state = [&]()
				{
					output->push_back(vehicle_events.get_current());
					output->push_back(eq.get_tot_iter());
					output->push_back(eq.get_rej_iter());
					output->push_back(eq.get_t());
					for (size_t i = 0; i < 7; i++) output->push_back(eq.get_y(i));
					for (size_t i = 3; i < 6; i++) output->push_back(eq.get_yp(i));
				};

			double tf = vehicle_events.get_tf();
			size_t steps = size_t(500 * (tf - t)) + 1;
			double dt = (tf - t) / steps;

			for (size_t step = 0; step <= steps; step++)
			{
				eq.stepn(t + step * dt, tf);
				eq.get_y(0, 7, y);

				if (r.norm() > switch_altitude)
				{
					for (size_t iter = 0; iter < 10; iter++)
					{
						double f = r.norm() - switch_altitude;
						double df = r.dot(v) / r.norm();
						eq.stepn(eq.get_t() - f / df);
						eq.get_y(0, 7, y);
						if (abs(f) < 1e-8)
						{
							t = eq.get_t();
							if (output) output_state();
							return;
						}
					}
					return; // refinement iterations exceeded
				}

				if (output) output_state();
				if (r.dot(v) < 0) return; // negative altitude reached
			}
			t = eq.get_t();
			vehicle_events.next();
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

		py_util::Spl spl_pressure = py_util::spline(py_params["splines"]["pressure"]);
		py_util::Spl spl_density = py_util::spline(py_params["splines"]["density"]);
		py_util::Spl spl_drag = py_util::spline(py_params["splines"]["drag"]);
		py_util::Spl spl_drag_mul = py_util::spline(py_params["splines"]["drag_mul"]);

		Eigen::Vector3d angular_velocity = py_util::vector3d(py_params["settings"]["angular_velocity"]);
		Eigen::Vector3d rdim_angular_velocity = scalar.rdim(Scalar::Quantity::RATE, 1.0) * angular_velocity;
		double initial_angle = py_params["settings"]["initial_angle"].cast<double>();

		double* rdim_output = py_rdim_output.mutable_data();
		for (size_t i = 0; i < int(output.size() / 14); i++)
		{
			for (size_t j = 0; j < 3; j++) rdim_output[26 * i + j] = output[14 * i + j];
			for (size_t j = 0; j < 11; j++) rdim_output[26 * i + j + 3] = scalar.rdim(scalar_quantities[j], output[14 * i + j + 3]);

			// rotation
			double t_output = rdim_output[26 * i + 3];
			double angle = initial_angle + t_output * rdim_angular_velocity.norm();
			Eigen::AngleAxisd rotation = Eigen::AngleAxisd(angle, rdim_angular_velocity.normalized());

			// state
			Eigen::Map<Eigen::Vector3d> r_output(&rdim_output[26 * i + 4]);
			Eigen::Map<Eigen::Vector3d> v_output(&rdim_output[26 * i + 7]);
			r_output = rotation * r_output;
			v_output = rotation * v_output + rdim_angular_velocity.cross(r_output);

			// accelerations
			Eigen::Map<Eigen::Vector3d> r(&output[14 * i + 4]);
			Eigen::Map<Eigen::Vector3d> v(&output[14 * i + 7]);
			double& m = output[14 * i + 10];
			Eigen::Map<Eigen::Vector3d> a(&output[14 * i + 11]);

			double rnorm = r.norm();
			double vnorm = v.norm();
			double rdim_rnorm = scalar.rdim(Scalar::Quantity::DISTANCE, rnorm);
			double rdim_vnorm = scalar.rdim(Scalar::Quantity::VELOCITY, vnorm);

			double pressure = exp(spl_pressure.eval(rdim_rnorm));
			double atmospheres = pressure / 101325.0;
			double density = exp(spl_density.eval(rdim_rnorm));
			double mach = rdim_vnorm / sqrt(1.4 * pressure / density);
			double drag_force = spl_drag.eval(mach) * spl_drag_mul.eval(density * rdim_vnorm) * density * rdim_vnorm * rdim_vnorm;

			Eigen::Map<Eigen::Vector3d> a_drag_output(&rdim_output[26 * i + 14]);
			Eigen::Map<Eigen::Vector3d> a_grav_output(&rdim_output[26 * i + 17]);
			Eigen::Map<Eigen::Vector3d> a_rot_output(&rdim_output[26 * i + 20]);
			Eigen::Map<Eigen::Vector3d> a_thrust_output(&rdim_output[26 * i + 23]);
			a_drag_output = -(rotation * v).normalized() * drag_force / scalar.rdim(Scalar::Quantity::MASS, m);
			a_grav_output = scalar.rdim(Scalar::Quantity::ACCELERATION, 1.0) * (rotation * -r) / pow(rnorm, 3.0);
			a_rot_output = scalar.rdim(Scalar::Quantity::ACCELERATION, 1.0) * (rotation * (-2.0 * angular_velocity.cross(v) - angular_velocity.cross(angular_velocity.cross(r))));
			a_thrust_output = scalar.rdim(Scalar::Quantity::ACCELERATION, 1.0) * (rotation * a) - a_drag_output - a_grav_output - a_rot_output;
		}
	}

	// atmospheric python functions

	py::tuple py_simulate_atm_phase(double t, py::array_t<double> py_yi, double pitch_rate, double azimuth, py::list py_events, py::dict py_params)
	{
		IntegratorParams integrator_params(py_params);
		AtmParams atm_params(pitch_rate, azimuth, py_params);
		VehicleEvents vehicle_events(py_events, py_params);

		double switch_altitude = py_params["settings"]["switch_altitude"].cast<double>();

		py::array_t<double> py_y;
		py_util::array_copy(py_yi, py_y);
		double* y = py_y.mutable_data();
		simulate_atm_phase(t, y, switch_altitude, vehicle_events, atm_params, integrator_params, nullptr);

		Eigen::Map<Eigen::Vector3d> r(y);
		Eigen::Map<Eigen::Vector3d> v(y + 3);
		double angle = py_params["settings"]["initial_angle"].cast<double>() + t * atm_params.angular_velocity.norm();
		Eigen::AngleAxisd rotation = Eigen::AngleAxisd(angle, atm_params.angular_velocity.normalized());
		r = rotation * r;
		v = rotation * v + atm_params.angular_velocity.cross(r);

		return py::make_tuple(t, py_y);
	}

	py::array_t<double> py_output_atm_phase(double t, py::array_t<double> py_yi, double pitch_rate, double azimuth, py::list py_events, py::dict py_params)
	{
		IntegratorParams integrator_params(py_params);
		AtmParams atm_params(pitch_rate, azimuth, py_params);
		VehicleEvents vehicle_events(py_events, py_params);

		double switch_altitude = py_params["settings"]["switch_altitude"].cast<double>();

		std::vector<double> output;
		output.reserve(1024);

		py::array_t<double> py_y;
		py_util::array_copy(py_yi, py_y);
		double* y = py_y.mutable_data();
		simulate_atm_phase(t, y, switch_altitude, vehicle_events, atm_params, integrator_params, &output);

		py::array_t<double> py_rdim_output({ int(output.size() / 14), 26 });
		rdim_atm_output(output, py_rdim_output, py_params);
		return py_rdim_output;
	}

	// vacuum functions
	static void vac_state_derivatives_passive_guidance(double t, double y[], double yp[], void* params)
	{
		// params
		VacParams* vac_params = reinterpret_cast<VacParams*>(params);

		// state
		Eigen::Map<Eigen::Vector3d> r(y);
		Eigen::Map<Eigen::Vector3d> v(y + 3);
		double& m = y[6];

		// state derivatives
		Eigen::Map<Eigen::Vector3d> dr(yp);
		Eigen::Map<Eigen::Vector3d> dv(yp + 3);
		double& dm = yp[6];

		// guidance
		double thrust = vac_params->const_accel ? m * vac_params->a_limit : vac_params->thrust;
		Eigen::Vector3d l = Eigen::Vector3d(v);

		// set derivatives
		dr = v;
		dv = -r / pow(r.norm(), 3.0) + l.normalized() * thrust / m;
		dm = thrust > 0 ? -thrust / vac_params->thrust * vac_params->mass_rate : 0.0;
	}

	static void vac_state_derivatives_active_guidance(double t, double y[], double yp[], void* params)
	{
		// params
		VacParams* vac_params = reinterpret_cast<VacParams*>(params);

		// state
		Eigen::Map<Eigen::Vector3d> r(y);
		Eigen::Map<Eigen::Vector3d> v(y + 3);
		double& m = y[6];

		// state derivatives
		Eigen::Map<Eigen::Vector3d> dr(yp);
		Eigen::Map<Eigen::Vector3d> dv(yp + 3);
		double& dm = yp[6];

		// guidance
		double dt = t - vac_params->tgi;
		double thrust = vac_params->const_accel ? m * vac_params->a_limit : vac_params->thrust;
		Eigen::Vector3d l = vac_params->li * cos(dt) + vac_params->ldi * sin(dt);

		// set derivatives
		dr = v;
		dv = -r / pow(r.norm(), 3.0) + l.normalized() * thrust / m;
		dm = thrust > 0 ? -thrust / vac_params->thrust * vac_params->mass_rate : 0.0;
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

		// set derivatives
		dr = v;
		dv = -r / pow(r.norm(), 3.0);
		dm = 0.0;
	}

	static void vac_state_derivatives_stm(double t, double y[], double yp[], void* params)
	{
		// params
		VacParams* vac_params = reinterpret_cast<VacParams*>(params);

		// state
		Eigen::Map<Eigen::Vector3d> r(y);
		Eigen::Map<Eigen::Vector3d> v(y + 3);
		double& m = y[6];

		// state derivatives
		Eigen::Map<Eigen::Vector3d> dr(yp);
		Eigen::Map<Eigen::Vector3d> dv(yp + 3);
		double& dm = yp[6];

		// state transition matrices
		Eigen::Map<Eigen::Matrix<double, 6, 6>> dx_dxi(y + 7);
		Eigen::Map<Eigen::Matrix<double, 6, 6>> dx_dLi(y + 43);
		Eigen::Map<Eigen::Matrix<double, 6, 1>> dx_dmi(y + 79);

		// state transition derivative matrices
		Eigen::Map<Eigen::Matrix<double, 6, 6>> dxd_dxi(yp + 7);
		Eigen::Map<Eigen::Matrix<double, 6, 6>> dxd_dLi(yp + 43);
		Eigen::Map<Eigen::Matrix<double, 6, 1>> dxd_dmi(yp + 79);

		// guidance
		double dt = t - vac_params->tgi;
		double thrust = vac_params->const_accel ? m * vac_params->a_limit : vac_params->thrust;
		Eigen::Vector3d l = vac_params->li * cos(dt) + vac_params->ldi * sin(dt);

		// set derivatives
		dr = v;
		dv = -r / pow(r.norm(), 3.0) + l.normalized() * thrust / m;
		dm = thrust > 0 ? -thrust / vac_params->thrust * vac_params->mass_rate : 0.0;

		// set state transition derivative matrices
		Eigen::Matrix<double, 6, 6> dxd_dx = Eigen::Matrix<double, 6, 6>::Zero();
		dxd_dx.topRightCorner<3, 3>().setIdentity();
		dxd_dx.bottomLeftCorner<3, 3>() = 3.0 * r * r.transpose() * pow(r.norm(), -5.0) - Eigen::Matrix3d::Identity() * pow(r.norm(), -3.0);

		dxd_dxi = dxd_dx * dx_dxi;

		Eigen::Matrix<double, 6, 3> dxd_dl = Eigen::Matrix<double, 6, 3>::Zero();
		dxd_dl.bottomRows<3>() = thrust / m * (-l * l.transpose() + Eigen::Matrix3d::Identity() * l.squaredNorm()) * pow(l.norm(), -3.0);

		Eigen::Matrix<double, 3, 6> dl_dLi;
		dl_dLi.leftCols<3>() = cos(dt) * Eigen::Matrix3d::Identity();
		dl_dLi.rightCols<3>() = sin(dt) * Eigen::Matrix3d::Identity();

		dxd_dLi = dxd_dx * dx_dLi + dxd_dl * dl_dLi;

		Eigen::Matrix<double, 6, 1> dxd_dm = Eigen::Matrix<double, 6, 1>::Zero();
		if (!vac_params->const_accel) dxd_dm.bottomRows<3>() = -l.normalized() * thrust / m / m;

		dxd_dmi = dxd_dx * dx_dmi + dxd_dm;
	}

	static void vac_state_derivatives_active_guidance_stm(double t, double y[], double yp[], void* params)
	{
		// params
		VacParams* vac_params = reinterpret_cast<VacParams*>(params);

		// state
		Eigen::Map<Eigen::Vector3d> r(y);
		Eigen::Map<Eigen::Vector3d> v(y + 3);
		double& m = y[6];

		// state derivatives
		Eigen::Map<Eigen::Vector3d> dr(yp);
		Eigen::Map<Eigen::Vector3d> dv(yp + 3);
		double& dm = yp[6];

		// state transition matrices
		Eigen::Map<Eigen::Matrix<double, 6, 6>> dx_dLi(y + 7);

		// state transition derivative matrices
		Eigen::Map<Eigen::Matrix<double, 6, 6>> dxd_dLi(yp + 7);

		// guidance
		double dt = t - vac_params->tgi;
		double thrust = vac_params->const_accel ? m * vac_params->a_limit : vac_params->thrust;
		Eigen::Vector3d l = vac_params->li * cos(dt) + vac_params->ldi * sin(dt);

		// set derivatives
		dr = v;
		dv = -r / pow(r.norm(), 3.0) + l.normalized() * thrust / m;
		dm = thrust > 0 ? -thrust / vac_params->thrust * vac_params->mass_rate : 0.0;

		// set state transition derivative matrices
		Eigen::Matrix<double, 6, 6> dxd_dx = Eigen::Matrix<double, 6, 6>::Zero();
		dxd_dx.topRightCorner<3, 3>().setIdentity();
		dxd_dx.bottomLeftCorner<3, 3>() = 3.0 * r * r.transpose() * pow(r.norm(), -5.0) - Eigen::Matrix3d::Identity() * pow(r.norm(), -3.0);

		Eigen::Matrix<double, 6, 3> dxd_dl = Eigen::Matrix<double, 6, 3>::Zero();
		dxd_dl.bottomRows<3>() = thrust / m * (-l * l.transpose() + Eigen::Matrix3d::Identity() * l.squaredNorm()) * pow(l.norm(), -3.0);

		Eigen::Matrix<double, 3, 6> dl_dLi;
		dl_dLi.leftCols<3>() = cos(dt) * Eigen::Matrix3d::Identity();
		dl_dLi.rightCols<3>() = sin(dt) * Eigen::Matrix3d::Identity();

		dxd_dLi = dxd_dx * dx_dLi + dxd_dl * dl_dLi;
	}

	static void vac_state_derivatives_coast_stm(double t, double y[], double yp[], void* params)
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
		Eigen::Map<Eigen::Matrix<double, 6, 6>> dx_dxi(y + 7);

		// state transition matrix derivatives
		Eigen::Map<Eigen::Matrix<double, 6, 6>> dxd_dxi(yp + 7);

		// set derivatives
		dr = v;
		dv = -r / pow(r.norm(), 3.0);
		dm = 0.0;

		// set state transition matrix derivatives
		Eigen::Matrix<double, 6, 6> dxd_dx = Eigen::Matrix<double, 6, 6>::Zero();
		dxd_dx.topRightCorner<3, 3>().setIdentity();
		dxd_dx.bottomLeftCorner<3, 3>() = 3.0 * r * r.transpose() * pow(r.norm(), -5.0) - Eigen::Matrix3d::Identity() * pow(r.norm(), -3.0);

		dxd_dxi = dxd_dx * dx_dxi;
	}

	static void simulate_vac_phase(void f(double t, double y[], double yp[], void* params), int neqn, double& t, double tout, double tf, double* y, double* yp, VehicleEvents* vehicle_events, VacParams* vac_params, IntegratorParams* integrator_params)
	{
		if (vehicle_events)
		{
			if (vehicle_events->get_stage()) y[6] = vehicle_events->get_mf();
			vac_params->const_accel = vehicle_events->get_const_accel();
			vac_params->thrust = vehicle_events->get_thrust_vac();
			vac_params->mass_rate = vehicle_events->get_mdot();
		}
		Equation eq = Equation(f, neqn, t, y, integrator_params->method, integrator_params->reltol, integrator_params->abstol, vac_params);
		eq.stepn(std::min(tf, tout), tf);
		t = eq.get_t();
		eq.get_y(0, neqn, y);
		if (yp) eq.get_yp(0, 7, yp);
	}

	static void simulate_vac_phase_output(void f(double t, double y[], double yp[], void* params), double& t, double tout, double tf, double* y, VehicleEvents* vehicle_events, VacParams* vac_params, IntegratorParams* integrator_params, std::vector<double>* output)
	{
		if (vehicle_events)
		{
			if (vehicle_events->get_stage()) y[6] = vehicle_events->get_mf();
			vac_params->const_accel = vehicle_events->get_const_accel();
			vac_params->thrust = vehicle_events->get_thrust_vac();
			vac_params->mass_rate = vehicle_events->get_mdot();
		}

		Equation eq = Equation(f, 7, t, y, integrator_params->method, integrator_params->reltol, integrator_params->abstol, vac_params);

		size_t steps = size_t(500 * (tf - t)) + 1;
		double dt = (tf - t) / steps;

		for (size_t step = 0; step <= steps; step++)
		{
			eq.stepn(std::min(t + step * dt, tout), tf);
			if (vehicle_events)
			{
				output->push_back(vehicle_events->get_current());
			}
			else
			{
				output->push_back(0.0);
			}
			output->push_back(eq.get_tot_iter());
			output->push_back(eq.get_rej_iter());
			output->push_back(eq.get_t());
			for (size_t i = 0; i < 7; i++) output->push_back(eq.get_y(i));
			for (size_t i = 3; i < 6; i++) output->push_back(eq.get_yp(i));
			if (t + step * dt > tout) break;
		}
		t = eq.get_t();
		eq.get_y(0, 7, y);
	}

	static void simulate_vac_phase_to_velocity(double& t, double* y, double final_velocity, VehicleEvents& vehicle_events, VacParams& vac_params, IntegratorParams& integrator_params)
	{
		Eigen::Map<Eigen::Vector3d> r(y);
		Eigen::Map<Eigen::Vector3d> v(y + 3);
		Eigen::Vector3d a;

		while (vehicle_events.get_tf() < t)
			vehicle_events.next();

		while (vehicle_events.is_valid())
		{
			if (vehicle_events.get_stage()) y[6] = vehicle_events.get_mf();
			vac_params.const_accel = vehicle_events.get_const_accel();
			vac_params.thrust = vehicle_events.get_thrust_vac();
			vac_params.mass_rate = vehicle_events.get_mdot();

			Equation eq = Equation(vac_state_derivatives_active_guidance, 7, t, y, integrator_params.method, integrator_params.reltol, integrator_params.abstol, &vac_params);

			double tf = vehicle_events.get_tf();
			size_t steps = size_t(500 * (tf - t)) + 1;
			double dt = (tf - t) / steps;

			for (size_t step = 0; step <= steps; step++)
			{
				eq.stepn(t + step * dt, tf);
				eq.get_y(0, 7, y);

				if (v.norm() > final_velocity)
				{
					for (size_t iter = 0; iter < 10; iter++)
					{
						eq.get_yp(3, 3, a.data());
						double f = v.norm() - final_velocity;
						double df = v.dot(a) / v.norm();
						eq.stepn(eq.get_t() - f / df);
						eq.get_y(0, 7, y);
						if (abs(f) < 1e-8)
						{
							t = eq.get_t();
							return;
						}
					}
					return; // refinement iterations exceeded
				}
			}
			t = eq.get_t();
			vehicle_events.next();
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
			Eigen::Map<Eigen::Vector3d> a(&output[14 * i + 11]);

			Eigen::Map<Eigen::Vector3d> a_drag_output(&rdim_output[26 * i + 14]);
			Eigen::Map<Eigen::Vector3d> a_grav_output(&rdim_output[26 * i + 17]);
			Eigen::Map<Eigen::Vector3d> a_rot_output(&rdim_output[26 * i + 20]);
			Eigen::Map<Eigen::Vector3d> a_thrust_output(&rdim_output[26 * i + 23]);

			a_drag_output = Eigen::Vector3d::Zero();
			a_grav_output = scalar.rdim(Scalar::Quantity::ACCELERATION, 1.0) * -r / pow(r.norm(), 3.0);
			a_rot_output = Eigen::Vector3d::Zero();
			a_thrust_output = scalar.rdim(Scalar::Quantity::ACCELERATION, 1.0) * a - a_grav_output;
		}
	}

	template <typename Derived>
	Derived cross(const Eigen::Vector3d& vector, const Eigen::MatrixBase<Derived>& matrix)
	{
		Eigen::Matrix3d skew{ {0.0, -vector(2), vector(1)}, {vector(2), 0.0, -vector(0)}, {-vector(1), vector(0), 0.0} };
		return skew * matrix;
	}

	template <typename Derived>
	Derived cross(const Eigen::MatrixBase<Derived>& matrix, const Eigen::Vector3d& vector)
	{
		Eigen::Matrix3d skew{ {0.0, -vector(2), vector(1)}, {vector(2), 0.0, -vector(0)}, {-vector(1), vector(0), 0.0} };
		return -skew * matrix;
	}

	// vacuum python functions
	py::tuple py_guidance_residuals_jacobian(double t, double coast_duration, py::array_t<double> py_yi, py::array_t<double> py_x, py::array_t<double> py_c, py::list py_events, py::dict py_params)
	{
		IntegratorParams integrator_params(py_params);
		VacParams vac_params(py_x, t, py_params);
		VehicleEvents vehicle_events(py_events, py_params);

		int mode = py_c.size() - 3;
		double dtg = py_x.at(6);
		double tgf = t + dtg;
		double* c = py_c.mutable_data();

		double y[43];
		double yp[7];
		std::copy(py_yi.mutable_data(), py_yi.mutable_data() + 7, y);
		Eigen::Map<Eigen::Matrix<double, 6, 6>> dx_dLi(y + 7);
		dx_dLi.setZero();

		// initial phase
		while (vehicle_events.get_tf() + coast_duration < t)
			vehicle_events.next();

		// guidance phase
		while (vehicle_events.is_valid())
		{
			double tf = vehicle_events.get_tf() + coast_duration;
			simulate_vac_phase(vac_state_derivatives_active_guidance_stm, 43, t, tgf, tf, y, yp, &vehicle_events, &vac_params, &integrator_params);
			if (tf > tgf) break;
			vehicle_events.next();
		}

		Eigen::Vector3d rf(y);
		Eigen::Vector3d vf(y + 3);
		Eigen::Vector3d lf = vac_params.li * cos(dtg) + vac_params.ldi * sin(dtg);
		Eigen::Vector3d ldf = -vac_params.li * sin(dtg) + vac_params.ldi * cos(dtg);
		Eigen::Vector3d sigma = rf.cross(ldf) - vf.cross(lf);

		Eigen::Matrix<double, 3, 7> dr_du;
		dr_du << dx_dLi.topRows<3>(), Eigen::Map<Eigen::Matrix<double, 3, 1>>(yp);

		Eigen::Matrix<double, 3, 7> dv_du;
		dv_du << dx_dLi.bottomRows<3>(), Eigen::Map<Eigen::Matrix<double, 3, 1>>(yp + 3);

		Eigen::Matrix<double, 3, 7> dlf_du;
		dlf_du << cos(dtg) * Eigen::Matrix3d::Identity(), sin(dtg) * Eigen::Matrix3d::Identity(), ldf;

		Eigen::Matrix<double, 3, 7> dldf_du;
		dldf_du << -sin(dtg) * Eigen::Matrix3d::Identity(), cos(dtg)* Eigen::Matrix3d::Identity(), -lf;

		Eigen::Matrix<double, 3, 7> dsigma_du = cross(dr_du, ldf) + cross(rf, dldf_du) - cross(dv_du, lf) - cross(vf, dlf_du);

		py::array_t<double> py_residuals;
		py_residuals.resize({ 7 });
		Eigen::Map<Eigen::Matrix<double, 7, 1>> residuals(py_residuals.mutable_data());

		py::array_t<double> py_dresiduals_du({ 7, 7 });
		Eigen::Map<Eigen::Matrix<double, 7, 7>> dresiduals_du(py_dresiduals_du.mutable_data());
	
		residuals(0) = rf.dot(rf) - c[0];
		residuals(1) = vf.dot(vf) - c[1];
		residuals(2) = rf.dot(vf) - c[2];
		residuals(6) = ldf.norm() - 1.0;

		dresiduals_du.col(0) = 2.0 * rf.transpose() * dr_du;
		dresiduals_du.col(1) = 2.0 * vf.transpose() * dv_du;
		dresiduals_du.col(2) = rf.transpose() * dv_du + vf.transpose() * dr_du;
		dresiduals_du.col(6) = ldf.transpose().normalized() * dldf_du;

		if (mode == 0)
		{
			residuals.middleRows<3>(3) = sigma;
			dresiduals_du.middleCols<3>(3) = dsigma_du.transpose();
		}
		else
		{
			Eigen::Vector3d k = -Eigen::Vector3d::UnitY();
			Eigen::Vector3d h = rf.cross(vf);
			Eigen::Matrix3d dr_dr = Eigen::Matrix3d::Identity();
			Eigen::Matrix3d dv_dv = Eigen::Matrix3d::Identity();
			Eigen::Matrix<double, 3, 7> dh_du = cross(dr_du, vf) + cross(rf, dv_du);
			residuals(3) = k.dot(h) - c[3];
			residuals(5) = sigma.dot(h);
			dresiduals_du.col(3) = k.transpose() * cross(dr_dr, vf) * dr_du + k.transpose() * cross(rf, dv_dv) * dv_du;
			dresiduals_du.col(5) = h.transpose() * dsigma_du + sigma.transpose() * dh_du;
			if (mode == 1)
			{
				residuals(4) = sigma.dot(k);
				dresiduals_du.col(4) = k.transpose() * dsigma_du;
			}
			else
			{
				Eigen::Vector3d n{ cos(c[4]), 0.0, sin(c[4]) };
				residuals(4) = n.dot(h);
				dresiduals_du.col(4) = n.transpose() * cross(dr_dr, vf) * dr_du + n.transpose() * cross(rf, dv_dv) * dv_du;
			}
		}	

		return py::make_tuple(py_residuals, py_dresiduals_du);
	}

	py::tuple py_simulate_vac_phase_to_velocity(double t, py::array_t<double> py_yi, py::array_t<double> py_x, double final_velocity, py::list py_events, py::dict py_params)
	{
		IntegratorParams integrator_params(py_params);
		VacParams vac_params(py_x, py_params);
		VehicleEvents vehicle_events(py_events, py_params);

		py::array_t<double> py_y;
		py_util::array_copy(py_yi, py_y);
		double* y = py_y.mutable_data();
		simulate_vac_phase_to_velocity(t, y, final_velocity, vehicle_events, vac_params, integrator_params);

		return py::make_tuple(t, py_y);
	}

	py::array_t<double> py_output_vac_phase(double t, py::array_t<double> py_yi, py::array_t<double> py_x, py::list py_events, py::dict py_params)
	{
		IntegratorParams integrator_params(py_params);
		VacParams vac_params(py_x, py_params);
		VehicleEvents vehicle_events(py_events, py_params);

		double dtg = py_x.at(6);
		double coast_time = py_x.at(7);
		double coast_duration = py_x.at(8);

		std::vector<double> output;
		output.reserve(1024);

		py::array_t<double> py_y;
		py_util::array_copy(py_yi, py_y);
		double* y = py_y.mutable_data();

		// initial phase
		while (vehicle_events.get_tf() < t)
			vehicle_events.next();

		while (vehicle_events.is_valid())
		{
			double tout = coast_time;
			double tf = vehicle_events.get_tf();
			simulate_vac_phase_output(vac_state_derivatives_passive_guidance, t, tout, tf, y, &vehicle_events, &vac_params, &integrator_params, &output);
			if (tf > tout) break;
			vehicle_events.next();
		}

		// coast phase
		double tout = coast_time + coast_duration;
		simulate_vac_phase_output(vac_state_derivatives_coast, t, tout, tout, y, nullptr, nullptr, &integrator_params, &output);

		// guidance phase
		while (vehicle_events.is_valid())
		{
			double tout = vac_params.tgi + dtg;
			double tf = vehicle_events.get_tf() + coast_duration;
			simulate_vac_phase_output(vac_state_derivatives_active_guidance, t, tout, tf, y, &vehicle_events, &vac_params, &integrator_params, &output);
			if (tf > tout) break;
			vehicle_events.next();
		}

		py::array_t<double> py_rdim_output({ int(output.size() / 14), 26 });
		rdim_vac_output(output, py_rdim_output, py_params);
		return py_rdim_output;
	}

	py::array_t<double> py_constraint_residuals(double t, py::array_t<double> py_yi, py::array_t<double> py_x, py::array_t<double> py_c, py::list py_events, py::dict py_params)
	{
		IntegratorParams integrator_params(py_params);
		VacParams vac_params(py_x, py_params);
		VehicleEvents vehicle_events(py_events, py_params);

		int mode = py_c.size() - 3;
		double dtg = py_x.at(6);
		double coast_time = py_x.at(7);
		double coast_duration = py_x.at(8);
		double* c = py_c.mutable_data();

		py::array_t<double> py_y;
		py_util::array_copy(py_yi, py_y);
		double* y = py_y.mutable_data();

		// initial phase
		while (vehicle_events.get_tf() < t)
			vehicle_events.next();

		while (vehicle_events.is_valid())
		{
			double tout = coast_time;
			double tf = vehicle_events.get_tf();
			simulate_vac_phase(vac_state_derivatives_passive_guidance, 7, t, tout, tf, y, nullptr, &vehicle_events, &vac_params, &integrator_params);
			if (tf > tout) break;
			vehicle_events.next();
		}

		// coast phase
		double tout = coast_time + coast_duration;
		simulate_vac_phase(vac_state_derivatives_coast, 7, t, tout, tout, y, nullptr, nullptr, nullptr, &integrator_params);

		// guidance phase
		while (vehicle_events.is_valid())
		{
			double tout = vac_params.tgi + dtg;
			double tf = vehicle_events.get_tf() + coast_duration;
			simulate_vac_phase(vac_state_derivatives_active_guidance, 7, t, tout, tf, y, nullptr, &vehicle_events, &vac_params, &integrator_params);
			if (tf > tout) break;
			vehicle_events.next();
		}

		Eigen::Vector3d rf(y);
		Eigen::Vector3d vf(y + 3);
		Eigen::Vector3d lf = vac_params.li * cos(dtg) + vac_params.ldi * sin(dtg);
		Eigen::Vector3d ldf = -vac_params.li * sin(dtg) + vac_params.ldi * cos(dtg);
		Eigen::Vector3d sigma = rf.cross(ldf) - vf.cross(lf);

		py::array_t<double> py_residuals;
		py_residuals.resize({ 7 });
		Eigen::Map<Eigen::Matrix<double, 7, 1>> residuals(py_residuals.mutable_data());

		residuals(0) = rf.dot(rf) - c[0];
		residuals(1) = vf.dot(vf) - c[1];
		residuals(2) = rf.dot(vf) - c[2];
		residuals(6) = ldf.norm() - 1.0;

		if (mode == 0)
		{
			residuals.middleRows<3>(3) = sigma;
		}
		else
		{
			Eigen::Vector3d k = -Eigen::Vector3d::UnitY();
			Eigen::Vector3d h = rf.cross(vf);
			residuals(3) = k.dot(h) - c[3];
			residuals(5) = sigma.dot(h);
			if (mode == 1)
			{
				residuals(4) = sigma.dot(k);
			}
			else
			{
				Eigen::Vector3d n{ cos(c[4]), 0.0, sin(c[4]) };
				residuals(4) = n.dot(h);
			}
		}

		return py_residuals;
	}
	
	py::array_t<double> py_constraint_jacobian(double t, py::array_t<double> py_yi, py::array_t<double> py_x, py::array_t<double> py_c, py::list py_events, py::dict py_params)
	{
		IntegratorParams integrator_params(py_params);
		VacParams vac_params(py_x, py_params);
		VehicleEvents vehicle_events(py_events, py_params);

		int mode = py_c.size() - 3;
		double dtg = py_x.at(6);
		double coast_time = py_x.at(7);
		double coast_duration = py_x.at(8);
		double* c = py_c.mutable_data();

		double y[7];
		double yc[43];
		double yg1[86];
		double ygn[86];

		double yp[7];
		double ypc[7];
		double ypg1[7];
		double ypgn[7];

		Eigen::Matrix<double, 6, 9> dx_du;

		// passive guidance phase
		std::copy(py_yi.mutable_data(), py_yi.mutable_data() + 7, y);
		while (vehicle_events.get_tf() < t)
			vehicle_events.next();

		while (vehicle_events.is_valid())
		{
			double tout = coast_time;
			double tf = vehicle_events.get_tf();
			simulate_vac_phase(vac_state_derivatives_passive_guidance, 7, t, tout, tf, y, yp, &vehicle_events, &vac_params, &integrator_params);
			if (tf > tout) break;
			vehicle_events.next();
		}

		// coast phase
		std::copy(y, y + 7, yc);
		Eigen::Map<Eigen::Matrix<double, 6, 6>> dx_dxi_c(yc + 7);
		dx_dxi_c.setIdentity();
		double tout = coast_time + coast_duration;
		simulate_vac_phase(vac_state_derivatives_coast_stm, 43, t, tout, tout, yc, ypc, nullptr, nullptr, &integrator_params);

		// guidance phase
		// remainder of coast stage
		std::copy(yc, yc + 7, yg1);
		Eigen::Map<Eigen::Matrix<double, 6, 6>> dx_dxi_g1(yg1 + 7);
		Eigen::Map<Eigen::Matrix<double, 6, 6>> dx_dLi_g1(yg1 + 43);
		Eigen::Map<Eigen::Matrix<double, 6, 1>> dx_dmi_g1(yg1 + 79);
		dx_dxi_g1.setIdentity();
		dx_dLi_g1.setZero();
		dx_dmi_g1.setZero();
		tout = vac_params.tgi + dtg;
		double tf = vehicle_events.get_tf() + coast_duration;
		simulate_vac_phase(vac_state_derivatives_stm, 86, t, tout, tf, yg1, ypg1, &vehicle_events, &vac_params, &integrator_params);

		if (tf > tout)
		{
			std::copy(yg1, yg1 + 7, y);
			dx_du.leftCols<6>() = dx_dLi_g1;
			dx_du.col(6) = Eigen::Map<Eigen::Matrix<double, 6, 1>>(ypg1);
			dx_du.col(7) = dx_dxi_g1 * dx_dxi_c * Eigen::Map<Eigen::Matrix<double, 6, 1>>(yp) + dx_dmi_g1 * yp[6];
			dx_du.col(8) = dx_dxi_g1 * Eigen::Map<Eigen::Matrix<double, 6, 1>>(ypc);
		}
		else
		{
			vehicle_events.next();
			std::copy(yg1, yg1 + 7, ygn);
			Eigen::Map<Eigen::Matrix<double, 6, 6>> dx_dxi_gn(ygn + 7);
			Eigen::Map<Eigen::Matrix<double, 6, 6>> dx_dLi_gn(ygn + 43);
			Eigen::Map<Eigen::Matrix<double, 6, 1>> dx_dmi_gn(ygn + 79);
			dx_dxi_gn.setIdentity();
			dx_dLi_gn.setZero();
			dx_dmi_gn.setZero();

			while (vehicle_events.is_valid())
			{
				double tf = vehicle_events.get_tf() + coast_duration;
				simulate_vac_phase(vac_state_derivatives_stm, 86, t, tout, tf, ygn, ypgn, &vehicle_events, &vac_params, &integrator_params);
				if (tf > tout) break;
				vehicle_events.next();
			}

			Eigen::Matrix<double, 6, 1> dL_dt;
			dL_dt << vac_params.ldi, -vac_params.li;

			std::copy(ygn, ygn + 7, y);
			dx_du.leftCols<6>() = dx_dxi_gn * dx_dLi_g1 + dx_dLi_gn;
			dx_du.col(6) = Eigen::Map<Eigen::Matrix<double, 6, 1>>(ypgn);
			dx_du.col(7) = dx_dxi_gn * (dx_dxi_g1 * dx_dxi_c * Eigen::Map<Eigen::Matrix<double, 6, 1>>(yp) + dx_dmi_g1 * yp[6] - Eigen::Map<Eigen::Matrix<double, 6, 1>>(ypg1)) + Eigen::Map<Eigen::Matrix<double, 6, 1>>(ypgn) - dx_dLi_gn * dL_dt;
			dx_du.col(8) = dx_dxi_gn * dx_dxi_g1 * Eigen::Map<Eigen::Matrix<double, 6, 1>>(ypc);
		}

		Eigen::Vector3d rf(y);
		Eigen::Vector3d vf(y + 3);
		Eigen::Vector3d lf = vac_params.li * cos(dtg) + vac_params.ldi * sin(dtg);
		Eigen::Vector3d ldf = -vac_params.li * sin(dtg) + vac_params.ldi * cos(dtg);
		Eigen::Vector3d sigma = rf.cross(ldf) - vf.cross(lf);

		Eigen::Matrix<double, 3, 9> dr_du = dx_du.topRows(3);

		Eigen::Matrix<double, 3, 9> dv_du = dx_du.bottomRows(3);

		Eigen::Matrix<double, 3, 9> dlf_du;
		dlf_du << cos(dtg) * Eigen::Matrix3d::Identity(), sin(dtg)* Eigen::Matrix3d::Identity(), ldf, 0.0, 0.0;

		Eigen::Matrix<double, 3, 9> dldf_du;
		dldf_du << -sin(dtg) * Eigen::Matrix3d::Identity(), cos(dtg)* Eigen::Matrix3d::Identity(), -lf, 0.0, 0.0;

		Eigen::Matrix<double, 3, 9> dsigma_du = cross(dr_du, ldf) + cross(rf, dldf_du) - cross(dv_du, lf) - cross(vf, dlf_du);

		py::array_t<double> py_dresiduals_du({ 7, 9 });
		Eigen::Map<Eigen::Matrix<double, 9, 7>> dresiduals_du(py_dresiduals_du.mutable_data());
		
		dresiduals_du.col(0) = 2.0 * rf.transpose() * dr_du;
		dresiduals_du.col(1) = 2.0 * vf.transpose() * dv_du;
		dresiduals_du.col(2) = rf.transpose() * dv_du + vf.transpose() * dr_du;
		dresiduals_du.col(6) = ldf.transpose().normalized() * dldf_du;

		if (mode == 0)
		{
			dresiduals_du.middleCols<3>(3) = dsigma_du.transpose();
		}
		else
		{
			Eigen::Vector3d k = -Eigen::Vector3d::UnitY();
			Eigen::Vector3d h = rf.cross(vf);
			Eigen::Matrix3d dr_dr = Eigen::Matrix3d::Identity();
			Eigen::Matrix3d dv_dv = Eigen::Matrix3d::Identity();
			Eigen::Matrix<double, 3, 9> dh_du = cross(dr_du, vf) + cross(rf, dv_du);
			dresiduals_du.col(3) = k.transpose() * cross(dr_dr, vf) * dr_du + k.transpose() * cross(rf, dv_dv) * dv_du;
			dresiduals_du.col(5) = h.transpose() * dsigma_du + sigma.transpose() * dh_du;
			if (mode == 1)
			{
				dresiduals_du.col(4) = k.transpose() * dsigma_du;
			}
			else
			{
				Eigen::Vector3d n{ cos(c[4]), 0.0, sin(c[4]) };
				dresiduals_du.col(4) = n.transpose() * cross(dr_dr, vf) * dr_du + n.transpose() * cross(rf, dv_dv) * dv_du;
			}
		}

		return py_dresiduals_du;
	}
}