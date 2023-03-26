#include <nlopt.hpp>
#include "Ephemeris.h"
#include "Orbit.h"
#include <ConicLunarFlightPlan.h>
#include "astrodynamics.h"
#include <iostream>
#include <iomanip>

void constraint_numerical_gradient(unsigned m, unsigned n, const double* x, double* grad, void* f_data,
    void(*func)(unsigned m, double* result, unsigned n, const double* x, double* grad, void* f_data))
{
    double eps = 1.49011611938476e-08;
    std::vector<double> base(m);
    std::vector<double> pert(m);
    std::vector<double> p(n);

    for (int i = 0; i < n; i++)
    {
        p[i] = x[i];
    }

    func(m, base.data(), n, x, NULL, f_data);

    double pold;

    for (int i = 0; i < n; i++)
    {
        if (i != 0)
        {
            p[i - 1] = pold;
        }
        pold = p[i];
        p[i] += eps;

        func(m, pert.data(), n, p.data(), NULL, f_data);

        for (int j = 0; j < m; j++)
        {
            grad[i + j * n] = (pert[j] - base[j]) / eps;
        }
    }
}

void objective_numerical_gradient(unsigned n, const double* x, double* grad,
    void* f_data, double(*func)(unsigned n, const double* x, double* grad, void* f_data))
{
    double eps = 1.49011611938476e-08;
    double base = func(n, x, NULL, f_data);
    std::vector<double> p(n);

    for (int i = 0; i < n; i++)
    {
        p[i] = x[i];
    }

    double pold;

    for (int i = 0; i < n; i++)
    {
        if (i != 0)
        {
            p[i - 1] = pold;
        }
        pold = p[i];
        p[i] += eps;

        grad[i] = (func(n, p.data(), NULL, f_data) - base) / eps;
    }
}

struct TCMData
{
    astrodynamics::ConicBody* moon;
    astrodynamics::ConicBody* planet;
    std::vector<double> t;
    Eigen::Vector3d r0;
    Eigen::Vector3d v0;
    Eigen::Vector3d rf;
    Eigen::Vector3d vf;
    double eps;
};

void constraints_tcm(unsigned m, double* result, unsigned n, const double* x, double* grad, void* f_data)
{
    TCMData* data = reinterpret_cast<TCMData*>(f_data);

    Eigen::Vector3d rp = data->r0;
    Eigen::Vector3d vp = data->v0;
    Eigen::Vector3d rm = data->rf - data->moon->orbit->get_position(data->t[data->t.size() - 1]);
    Eigen::Vector3d vm = data->vf - data->moon->orbit->get_velocity(data->t[data->t.size() - 1]);

    double tsoi;

    // within moon soi
    for (int i = data->t.size() - 1; i > 0; i--)
    {
        vm(0) -= x[3 * (i - 1) + 0];
        vm(1) -= x[3 * (i - 1) + 1];
        vm(2) -= x[3 * (i - 1) + 2];

        double t0 = data->t[i - 1];
        double t1 = data->t[i];
        double dtsoi = astrodynamics::hyperbolic_orbit_time_at_distance(rm, vm, data->moon->soi, data->moon->mu) - astrodynamics::hyperbolic_orbit_time_at_distance(rm, vm, rm.norm(), data->moon->mu);

        if (t1 - t0 > dtsoi)
        {
            tsoi = t1 - dtsoi;
            std::tie(rm, vm) = astrodynamics::kepler_s(rm, vm, tsoi - t1, data->moon->mu, data->eps);
            break;
        }

        std::tie(rm, vm) = astrodynamics::kepler_s(rm, vm, t0 - t1, data->moon->mu, data->eps);
    }

    // within planet soi
    for (int i = 0; i < data->t.size(); i++)
    {
        double t0 = data->t[i];
        double t1 = data->t[i + 1];

        if (t1 > tsoi)
        {
            t1 = tsoi;
            std::tie(rp, vp) = astrodynamics::kepler_s(rp, vp, tsoi - t0, data->planet->mu, data->eps);
            break;
        }

        std::tie(rp, vp) = astrodynamics::kepler_s(rp, vp, t1 - t0, data->planet->mu, data->eps);
        vp(0) += x[3 * i + 0];
        vp(1) += x[3 * i + 1];
        vp(2) += x[3 * i + 2];
    }

    Eigen::Vector3d dr = rp - (rm + data->moon->orbit->get_position(tsoi));
    Eigen::Vector3d dv = vp - (vm + data->moon->orbit->get_velocity(tsoi));

    result[0] = dr(0) / data->moon->orbit->get_distance_scale();
    result[1] = dr(1) / data->moon->orbit->get_distance_scale();
    result[2] = dr(2) / data->moon->orbit->get_distance_scale();
    result[3] = dv(0) / data->moon->orbit->get_velocity_scale();
    result[4] = dv(1) / data->moon->orbit->get_velocity_scale();
    result[5] = dv(2) / data->moon->orbit->get_velocity_scale();

    if (grad)
    {
        constraint_numerical_gradient(m, n, x, grad, f_data, constraints_tcm);
    }
}

double objective_tcm(unsigned n, const double* x, double* grad, void* f_data)
{
    TCMData* data = reinterpret_cast<TCMData*>(f_data);

    double f = 0.0;
    for (int i = 0; i < n / 3; i++)
    {
        f += sqrt(
            x[3 * i + 0] * x[3 * i + 0] +
            x[3 * i + 1] * x[3 * i + 1] +
            x[3 * i + 2] * x[3 * i + 2]
        );
    }

    if (grad)
    {
        objective_numerical_gradient(n, x, grad, f_data, objective_tcm);
    }

    return f;
}

int main()
{

	Orbit planet_orbit = Orbit(
		1.1723328e18,		// gravitational_parameter
		1.3599840256e10,	// semi_major_axis
		0.0,				// eccentricity
		0.0,				// inclination
		0.0,				// longitude_of_ascending_node
		0.0,				// argument_of_periapsis
		3.14,				// mean_anomaly_at_epoch
		0.0);				// epoch

	Orbit moon_orbit = Orbit(
		3.5316e12,	// gravitational_parameter
		1.2e7,		// semi_major_axis
		0.0,		// eccentricity
		0.0,		// inclination
		0.0,		// longitude_of_ascending_node
		0.0,		// argument_of_periapsis
		1.7,		// mean_anomaly_at_epoch
		0.0);		// epoch

	astrodynamics::ConicBody planet = { &planet_orbit, 3.5316e12, 84159286.0, 6.0e5 };
	astrodynamics::ConicBody moon = { &moon_orbit, 6.5138398e10, 2.4295591e6, 2.0e5 };


    try
    {
        TCMData tcm_data;

        tcm_data.moon = &moon;
        tcm_data.planet = &planet;

        tcm_data.r0 << -239637.56293582544, 324.1700200737195, 678254.4980235416;
        tcm_data.v0 << -3030.693848690596, 0.16076759442235752, -334.76234161450003;
        tcm_data.rf << -2385264.830117757, -2375.5891111876012, -12186383.898833545;
        tcm_data.vf << 121.5966634293062, 1.8945879498592202, 313.45528750092;
        tcm_data.t = { 34923.49952389345, 35043.49952389345, 42123.49952389345, 62869.90541117803 };

        tcm_data.eps = 1e-10;

        int m = 6;
        int n = (tcm_data.t.size() - 1) * 3;
        std::vector<double> x0(n, tcm_data.eps);

        std::vector<double> tol(m, tcm_data.eps);
        double minf;

        nlopt::opt opt = nlopt::opt("LD_SLSQP", n);
        opt.set_min_objective(objective_tcm, &tcm_data);
        opt.add_equality_mconstraint(constraints_tcm, &tcm_data, tol);
        opt.set_ftol_abs(tcm_data.eps);

        try
        {
            opt.optimize(x0, minf);
        }
        catch (const std::exception& e)
        {
            std::cerr << "Error during optimization: " << e.what() << '\n';
        }

        std::cout << std::setprecision(17);
        std::cout << "last optimum value: " << opt.last_optimum_value() << '\n';
        std::cout << "last optimum result: " << opt.last_optimize_result() << '\n';
        std::cout << "num evals: " << opt.get_numevals() << '\n';

        std::vector<double> result(6);

        constraints_tcm(m, result.data(), n, x0.data(), NULL, &tcm_data);
    }
    catch (std::exception& e)
    {
        std::cerr << e.what() << "\n";
    }

	// Ephemeris eph = Ephemeris("earth_eph823");
	// LunarFlightPlan lfp(eph);

	ConicLunarFlightPlan clfp(planet, moon);
	clfp.set_mission(100000.0, ConicLunarFlightPlan::TrajectoryMode::FREE_RETURN, 680000.0, 214000.0, 1.2);





	//void set_mission(Jdate initial_time, bool free_return, double rp_earth, double rp_moon, double* initial_orbit = nullptr);
	//void add_min_flight_time_constraint(double min_time);
	//void add_max_flight_time_constraint(double max_time);
	//void add_inclination_constraint(bool launch, double min, double max, Eigen::Vector3d n);

	// lfp.set_mission(Jdate(1, 2, 1951), LunarFlightPlan::TrajectoryMode::RETURN, 6371000.0 + 250000.0, 1731100.0 + 170000.0, 1.3);
	// lfp.add_min_flight_time_constraint(2.0);
	// lfp.add_max_flight_time_constraint(2.9);
	// lfp.add_inclination_constraint(true, 28.0, 70.0, Eigen::Vector3d::UnitY());

	// lfp.init_model();
	try
	{
		// lfp.run_model(2000, 1e-8, 1e-8, 1e-8);
		// LunarFlightPlan::Result res = lfp.output_result(1e-8);

		clfp.init_model();
		// clfp.run_model(2000, 1e-8, 1e-8, 1e-8);
		ConicLunarFlightPlan::Result res = clfp.output_result(1e-8);

		double minc = 0.0;
		double maxc = 0.0;

		for (int i = 0; i < res.r[0].size(); i++)
		{
			std::cout << std::setprecision(17)
				<< res.r[0][i] << ", "
				//<< res.r[1][i] << ", "
				<< res.r[2][i] << "\n";
		}

		/*		for (auto& c : res.nlopt_constraints)
				{
					if (c < minc) minc = c;
					if (c > maxc) maxc = c;
				}*/

		std::cout << "minc: " << minc << '\n';
		std::cout << "maxc: " << maxc << '\n';
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << '\n';
		ConicLunarFlightPlan::Result res = clfp.output_result(1e-8);
		for (int i = 0; i < res.r[0].size(); i++)
		{
			std::cout << std::setprecision(17)
				<< res.r[0][i] << ", "
				<< res.r[1][i] << ", "
				<< res.r[2][i] << "\n";
		}
		throw e;
	}


	return 0;
}



//#include <nlopt.hpp>
//#include "Ephemeris.h"
//#include <LunarFlightPlan.h>
//
//#include <iostream>
//
//int main()
//{
//	Ephemeris eph = Ephemeris("earth_eph823");
//	LunarFlightPlan lfp(eph);
//
//	//void set_mission(Jdate initial_time, bool free_return, double rp_earth, double rp_moon, double* initial_orbit = nullptr);
//	//void add_min_flight_time_constraint(double min_time);
//	//void add_max_flight_time_constraint(double max_time);
//	//void add_inclination_constraint(bool launch, double min, double max, Eigen::Vector3d n);
//
//	lfp.set_mission(Jdate(1, 2, 1951), LunarFlightPlan::TrajectoryMode::RETURN, 6371000.0 + 250000.0, 1731100.0 + 170000.0, 1.3);
//	// lfp.add_min_flight_time_constraint(2.0);
//	// lfp.add_max_flight_time_constraint(2.9);
//	// lfp.add_inclination_constraint(true, 28.0, 70.0, Eigen::Vector3d::UnitY());
//
//	lfp.init_model();
//	try
//	{
//		lfp.run_model(2000, 1e-8, 1e-8, 1e-8);
//		LunarFlightPlan::Result res = lfp.output_result(1e-8);
//
//		double minc = 0.0;
//		double maxc = 0.0;
//		for (auto& c : res.nlopt_constraints)
//		{
//			if (c < minc) minc = c;
//			if (c > maxc) maxc = c;
//		}
//
//		std::cout << "minc: " << minc << '\n';
//		std::cout << "maxc: " << maxc << '\n';
//	}
//	catch (const std::exception& e)
//	{
//		std::cerr << e.what() << '\n';
//		throw e;
//	}
//
//
//	return 0;
//}