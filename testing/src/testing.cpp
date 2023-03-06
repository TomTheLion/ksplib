#include <nlopt.hpp>
#include "Ephemeris.h"
#include "Orbit.h"
#include <ConicLunarFlightPlan.h>
#include "astrodynamics.h"
#include <iostream>
#include <iomanip>

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

	astrodynamics::ConicBody moon = { &moon_orbit, 6.5138398e10, 2.4295591e6, 2.0e5 };
	astrodynamics::ConicBody planet = { &planet_orbit, 3.5316e12, 8.4159287, 6.0e5 };




	// Ephemeris eph = Ephemeris("earth_eph823");
	// LunarFlightPlan lfp(eph);

	ConicLunarFlightPlan clfp(planet, moon);
	clfp.set_mission(0.0, ConicLunarFlightPlan::TrajectoryMode::LEAVE, 680000.0, 214000.0, 1.2);





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
		clfp.run_model(2000, 1e-8, 1e-8, 1e-8);
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