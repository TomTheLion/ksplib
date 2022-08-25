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

	// lfp.set_mission(Jdate(1, 2, 1951), true, 6371000.0 + 250000.0, 1731100.0 + 70000.0);
	std::vector<double> initial_orbit = { 6371000.0 + 250000.0, 0.0, 10.0, 0.0, 10.0, 7800.0 };
	lfp.set_mission(Jdate(1, 2, 1951), true, 6371000.0 + 250000.0, 1731100.0 + 70000.0);

	lfp.init_model();
	try
	{
		lfp.run_model(1000, 1e-8, 1e-8, 1e-14);
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << '\n';
		throw e;
	}
	

	return 0;
}