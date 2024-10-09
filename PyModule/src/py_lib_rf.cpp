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
	Scalar::Quantity state_scalars[] = {
		Scalar::Quantity::DISTANCE,
		Scalar::Quantity::DISTANCE,
		Scalar::Quantity::DISTANCE,
		Scalar::Quantity::VELOCITY,
		Scalar::Quantity::VELOCITY,
		Scalar::Quantity::VELOCITY,
		Scalar::Quantity::TIME };

	static void yp_vac(double t, double y[], double yp[], void* params)
	{

	}

	static void yp_vac_stm(double t, double y[], double yp[], void* params)
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
		double* p = reinterpret_cast<double*>(params);
		Eigen::Map<Eigen::Vector3d> li(p);
		Eigen::Map<Eigen::Vector3d> dli(p + 3);
		double& thrust = p[6];
		double& mass_rate = p[7];
		double& tg = p[8];
		double& a_limit = p[9];

		// guidance and a limit
		double dt = t - tg;
		bool a_limit_flag = a_limit && a_limit < thrust / m;
		double throttle = a_limit_flag ? m * a_limit / thrust : 1.0;
		Eigen::Vector3d l = li * cos(dt) + dli * sin(dt);

		// set derivatives
		dr = v;
		dv = -r / pow(r.norm(), 3.0) + throttle * l.normalized() * thrust / m;
		dm = -throttle * mass_rate;

		// set stm derivatives
		dstm.setZero();

		dr_dv.setIdentity();
		dv_dr = 3.0 * r * r.transpose() * pow(r.norm(), -5.0) - Eigen::Matrix3d::Identity() * pow(r.norm(), -3.0);
		dv_dm = l.normalized() * (a_limit_flag ? 0.0 : -thrust / m / m);
		dm_dm << (a_limit_flag ? -mass_rate * a_limit / thrust : 0.0);
		dv_dl = throttle * thrust / m * (-l * l.transpose() + Eigen::Matrix3d::Identity() * l.squaredNorm()) * pow(l.norm(), -3.0);

		dl_dld.setIdentity();
		dld_dl.setIdentity();
		dld_dl = -dld_dl;

		dstm *= stm;
	}


	static void simulate_vac_phase(double& t, py::array_t<double>& py_y, py::list& py_events, py::dict& py_params, std::vector<double>* output)
	{
		Scalar scalar = Scalar(
			py_params["scalars"]["time"].cast<double>(),
			py_params["scalars"]["distance"].cast<double>(),
			py_params["scalars"]["mass"].cast<double>());

		double* u = py_util::array_ptr(py_params["settings"]["u"]);

		double params[10] = {
			u[0], u[1], u[2], u[3], u[4], u[5], 0.0, 0.0,
			scalar.ndim(Scalar::Quantity::TIME, py_params["settings"]["tg"].cast<double>()),
			scalar.ndim(Scalar::Quantity::ACCELERATION, py_params["settings"]["a_limit"].cast<double>())
		};

		double& ndim_thrust_vac = params[6];
		double& ndim_mass_rate = params[7];

		std::string integrator = py_params["settings"]["integrator"].cast<std::string>();
		double reltol = py_params["settings"]["reltol"].cast<double>();
		double abstol = py_params["settings"]["abstol"].cast<double>();

		bool last = false;
		double* y = py_util::array_ptr(py_y);
		scalar.ndim(state_scalars, y, 7);
		double ndim_coast_time = scalar.ndim(Scalar::Quantity::TIME, py_params["settings"]["coast_time"].cast<double>());
		double ndim_coast_duration = scalar.ndim(Scalar::Quantity::TIME, py_params["settings"]["coast_duration"].cast<double>());
		double ndim_tf = u[6];

		for (size_t i = 0; i < py_events.size(); i++)
		{
			py::tuple py_event = py_events[i];
			ndim_thrust_vac = scalar.ndim(Scalar::Quantity::FORCE, py_event.attr("vacuum_thrust").cast<double>());
			ndim_mass_rate = scalar.ndim(Scalar::Quantity::MASS_RATE, py_event.attr("mass_rate").cast<double>());

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

			Equation eq = Equation(yp_vac, 7, t, y, integrator, reltol, abstol, &params);

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