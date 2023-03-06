#pragma once

#include <Eigen/Dense>

class Orbit
{
public:
	Orbit();

	Orbit(
		double gravitational_parameter,
		double semi_major_axis,
		double eccentricity,
		double inclination,
		double longitude_of_ascending_node,
		double argument_of_periapsis,
		double mean_anomaly_at_epoch,
		double epoch
	);

	Orbit(const Orbit& orbit);

	~Orbit();

	double get_distance_scale();
	double get_velocity_scale();
	double get_time_scale();

	Eigen::Vector3d get_position(double t);
	Eigen::Vector3d get_velocity(double t);
	Eigen::Vector3d get_acceleration(double t);

private:
	double distance_scale_;
	double velocity_scale_;
	double time_scale_;

	double gravitational_parameter_;
	double eccentricity_;
	double mean_anomaly_at_epoch_;
	double epoch_;

	double r_coef_;
	double v_coef_;
	double mean_angular_motion_;
	Eigen::Matrix3d m_;

	double last_t_;
	double last_theta_;

	double get_true_anomaly(double t);
	Eigen::Vector3d swap_yz(Eigen::Vector3d a);
};