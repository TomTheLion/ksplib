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

	static Spl spline(py::object obj)
	{
		py::tuple tup = obj;
		return Spl(array_ptr(tup[0]), array_ptr(tup[1]), array_size(tup[0]), tup[2].cast<size_t>());
	}
}

// if we could pass the base vehicle from python, and add the coast stuff here and calculate the gradient...
// create different derivative functions for vac phase depending on if using lambda vs velocity for the thrust direction
namespace kerbal_guidance_system
{
	struct VacParams {
		double ndim_vacuum_thrust;
		double ndim_mass_rate;
		double ndim_acceleration_limit;
		double ndim_tg;
		double* u;
	};

	static void yp_vac(double t, double y[], double yp[], void* params)
	{
		VacParams* vac_params = reinterpret_cast<VacParams*>(params);
		Eigen::Map<Eigen::Vector3d> ndim_r(y);
		Eigen::Map<Eigen::Vector3d> ndim_v(y + 3);
		Eigen::Map<Eigen::Vector3d> ndim_dr(yp);
		Eigen::Map<Eigen::Vector3d> ndim_dv(yp + 3);
		Eigen::Map<Eigen::Vector3d> li(vac_params->u);
		Eigen::Map<Eigen::Vector3d> dli(vac_params->u + 3);
		double ndim_m = y[6];
		double ndim_dm = yp[6];

		double dt = t - vac_params->ndim_tg;
		Eigen::Vector3d l = dt > 0 ? li * cos(dt) + dli * sin(dt) : Eigen::Vector3d(ndim_v);

		double ndim_thrust_acceleration = vac_params->ndim_vacuum_thrust / ndim_m;
		double throttle = ndim_thrust_acceleration && vac_params->ndim_acceleration_limit < ndim_thrust_acceleration ? vac_params->ndim_acceleration_limit / ndim_thrust_acceleration : 1.0;

		ndim_dr = ndim_v;
		ndim_dv = throttle * ndim_thrust_acceleration * l.normalized() - ndim_r / pow(ndim_r.norm(), 3.0);
		ndim_dm = -throttle * vac_params->ndim_mass_rate;
	}

	static void yp_vac_stm(double t, double y[], double yp[], void* params)
	{
		// state
		Eigen::Map<Eigen::Vector3d> r(y);
		Eigen::Map<Eigen::Vector3d> v(y + 3);
		Eigen::Map<Eigen::Vector3d> l(y + 6);
		Eigen::Map<Eigen::Vector3d> ld(y + 9);
		double m = y[12];
		double ft = y[13];
		double ve = y[14];
		Eigen::Map<Eigen::Matrix<double, 13, 13>> stm(y + 15);

		// derivatives
		Eigen::Map<Eigen::Vector3d> dr(yp);
		Eigen::Map<Eigen::Vector3d> dv(yp + 3);
		Eigen::Map<Eigen::Vector3d> dl(yp + 6);
		Eigen::Map<Eigen::Vector3d> dld(yp + 9);
		double dm = yp[12];
		double dft = yp[13];
		double dve = yp[14];
		Eigen::Map<Eigen::Matrix<double, 13, 13>> dstm(yp + 15);
		Eigen::Map<Eigen::Matrix3d> drdv(dstm.data(), 0, 3);

		// set derivatives
		dr = v;
		dv = -r / pow(r.norm(), 3.0) + l.normalized() * ft / m;
		dl = ld;
		dld = -l;
		dm = -ft / ve;
		dft = 0.0;
		dve = 0.0;
		
		// set stm derivatives
		dstm.setZero();
		drdv(0, 0) = 1.0;
		drdv(1, 1) = 1.0;
		drdv(2, 2) = 1.0;
	}


	static void simulate_vac_phase(double& t, py::array_t<double>& py_y, py::list& py_events, py::dict& py_params, std::vector<double>* output)
	{
		Scalar scalar = Scalar(
			py_params["scalars"]["time"].cast<double>(),
			py_params["scalars"]["distance"].cast<double>(),
			py_params["scalars"]["mass"].cast<double>());

		VacParams vac_params = VacParams{
			0.0,
			0.0,
			scalar.ndim(Scalar::Quantity::ACCELERATION, py_params["settings"]["a_limit"].cast<double>()),
			scalar.ndim(Scalar::Quantity::TIME, py_params["settings"]["tg"].cast<double>()),
			py_util::array_ptr(py_params["settings"]["u"])};

		std::string integrator = py_params["settings"]["integrator"].cast<std::string>();
		double reltol = py_params["settings"]["reltol"].cast<double>();
		double abstol = py_params["settings"]["abstol"].cast<double>();

		Scalar::Quantity state_scalars[] = {
			Scalar::Quantity::DISTANCE, 
			Scalar::Quantity::DISTANCE, 
			Scalar::Quantity::DISTANCE, 
			Scalar::Quantity::VELOCITY, 
			Scalar::Quantity::VELOCITY, 
			Scalar::Quantity::VELOCITY, 
			Scalar::Quantity::TIME };

		bool last = false;
		double* y = py_util::array_ptr(py_y);
		scalar.ndim(state_scalars, y, 7);
		double ndim_coast_time = scalar.ndim(Scalar::Quantity::TIME, py_params["settings"]["coast_time"].cast<double>());
		double ndim_coast_duration = scalar.ndim(Scalar::Quantity::TIME, py_params["settings"]["coast_duration"].cast<double>());
		double ndim_tf = vac_params.u[6];

		for (size_t i = 0; i < py_events.size(); i++)
		{
			py::tuple py_event = py_events[i];
			vac_params.ndim_vacuum_thrust = scalar.ndim(Scalar::Quantity::FORCE, py_event.attr("vacuum_thrust").cast<double>());
			vac_params.ndim_mass_rate = scalar.ndim(Scalar::Quantity::MASS_RATE, py_event.attr("mass_rate").cast<double>());

			bool stage = py_event.attr("stage").cast<bool>();
			double ndim_tout = scalar.ndim(Scalar::Quantity::TIME, py_event.attr("tout").cast<double>());

			// if tout > coast time
			// integrate to coast time
			// coast to coast_time + coast_duration
			// increase future tout values by coast duration

			if (ndim_tout < t) continue;
			if (stage)
			{
				y[6] = scalar.ndim(Scalar::Quantity::MASS, py_event.attr("mass_final").cast<double>());
			}

			Equation eq = Equation(yp_vac, 7, t, y, integrator, reltol, abstol, &vac_params);

			if (ndim_tout > ndim_tf)
			{
				ndim_tout = ndim_tf;
				last = true;
			}

			eq.step(ndim_tout);
			eq.get_y(0, 7, y);

			if (last)
			{
				scalar.rdim(state_scalars, y, 7);
				return;
			}
		}
	}

	static py::tuple py_simulate_vac_phase(double t, py::array_t<double> py_y, py::list py_events, py::dict py_params)
	{

		return py::make_tuple();
	}
}