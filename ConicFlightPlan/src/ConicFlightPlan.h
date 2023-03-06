#pragma once

struct Body
{
	double semi_major_axis;
	double eccentricity;
	double inclination;
	double longitude_of_ascending_node;
	double argument_of_periapsis;
	double mean_anomaly_at_epoch;
	double epoch;
	double sphere_of_influence;
};

class Trajectory
{
	Trajectory();

	~Trajectory();

	void add_body(Body body, double t, double radius);

	// add constraints for moons

};