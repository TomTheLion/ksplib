#include <nlopt.hpp>
#include "Ephemeris.h"
#include <LunarFlightPlan.h>

#include <iostream>

int main()
{
	Ephemeris eph = Ephemeris("earth_eph823");
	LunarFlightPlan lfp(eph);

	//void set_mission(Jdate initial_time, bool free_return, double rp_earth, double rp_moon, double* initial_orbit = nullptr);
	//void add_min_flight_time_constraint(double min_time);
	//void add_max_flight_time_constraint(double max_time);
	//void add_inclination_constraint(bool launch, double min, double max, Eigen::Vector3d n);

	lfp.set_mission(Jdate(1, 2, 1951), LunarFlightPlan::TrajectoryMode::RETURN, 6371000.0 + 250000.0, 1731100.0 + 170000.0, 1.3);
	// lfp.add_min_flight_time_constraint(2.0);
	// lfp.add_max_flight_time_constraint(2.9);
	// lfp.add_inclination_constraint(true, 28.0, 70.0, Eigen::Vector3d::UnitY());

	lfp.init_model();
	try
	{
		lfp.run_model(2000, 1e-8, 1e-8, 1e-8);
		LunarFlightPlan::Result res = lfp.output_result(1e-8);

		double minc = 0.0;
		double maxc = 0.0;
		for (auto& c : res.nlopt_constraints)
		{
			if (c < minc) minc = c;
			if (c > maxc) maxc = c;
		}	

		std::cout << "minc: " << minc << '\n';
		std::cout << "maxc: " << maxc << '\n';
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << '\n';
		throw e;
	}
	

	return 0;
}