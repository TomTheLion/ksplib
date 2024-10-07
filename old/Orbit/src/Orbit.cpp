#include "Orbit.h"

Orbit::Orbit()
{

}

Orbit::Orbit(
	double gravitational_parameter,
	double semi_major_axis,
	double eccentricity,
	double inclination,
	double longitude_of_ascending_node,
	double argument_of_periapsis,
	double mean_anomaly_at_epoch,
	double epoch)
	:
	gravitational_parameter_(gravitational_parameter),
	eccentricity_(eccentricity),
	mean_anomaly_at_epoch_(mean_anomaly_at_epoch),
	epoch_(epoch)
{
	distance_scale_ = semi_major_axis;
	time_scale_ = pow(semi_major_axis, 1.5) / sqrt(gravitational_parameter);
	velocity_scale_ = distance_scale_ / time_scale_;

	r_coef_ = semi_major_axis * (1.0 - eccentricity * eccentricity);
	v_coef_ = sqrt(gravitational_parameter / r_coef_);

	mean_angular_motion_ = pow(semi_major_axis, -1.5) * sqrt(gravitational_parameter);

	Eigen::Matrix3d m1, m2, m3;

	m1 <<
		cos(argument_of_periapsis), sin(argument_of_periapsis), 0.0,
		-sin(argument_of_periapsis), cos(argument_of_periapsis), 0.0,
		0.0, 0.0, 1.0;

	m2 <<
		1.0, 0.0, 0.0,
		0.0, cos(inclination), sin(inclination),
		0.0, -sin(inclination), cos(inclination);

	m3 <<
		cos(longitude_of_ascending_node), sin(longitude_of_ascending_node), 0.0,
		-sin(longitude_of_ascending_node), cos(longitude_of_ascending_node), 0.0,
		0.0, 0.0, 1.0;

	m_ = (m1 * m2 * m3).transpose();

	get_true_anomaly(0.0);
}

Orbit::Orbit(const Orbit& orbit)
{
	distance_scale_ = orbit.distance_scale_;
	velocity_scale_ = orbit.velocity_scale_;
	time_scale_ = orbit.time_scale_;

	gravitational_parameter_ = orbit.gravitational_parameter_;
	eccentricity_ = orbit.eccentricity_;
	mean_anomaly_at_epoch_ = orbit.mean_anomaly_at_epoch_;
	epoch_ = orbit.epoch_;

	r_coef_ = orbit.r_coef_;
	v_coef_ = orbit.v_coef_;
	mean_angular_motion_ = orbit.mean_angular_motion_;
	m_ = orbit.m_;

	last_t_ = orbit.last_t_;
	last_theta_ = orbit.last_theta_;
}

Orbit::~Orbit()
{

}

double Orbit::get_distance_scale()
{
	return distance_scale_;
}

double Orbit::get_velocity_scale()
{
	return velocity_scale_;
}

double Orbit::get_time_scale()
{
	return time_scale_;
}


Eigen::Vector3d Orbit::get_position(double t)
{
	double theta = t == last_t_ ? last_theta_ : get_true_anomaly(t);
	Eigen::Vector3d r;

	r << cos(theta), sin(theta), 0.0;

	return swap_yz(m_ * r_coef_ * 1.0 / (1.0 + eccentricity_ * cos(theta)) * r);
}

Eigen::Vector3d Orbit::get_velocity(double t)
{
	double theta = t == last_t_ ? last_theta_ : get_true_anomaly(t);
	Eigen::Vector3d v;

	v << -sin(theta), eccentricity_ + cos(theta), 0.0;

	return swap_yz(m_ * v_coef_ * v);
}

Eigen::Vector3d Orbit::get_acceleration(double t)
{
	Eigen::Vector3d r = get_position(t);

	return -gravitational_parameter_ / pow(r.norm(), 3.0) * r;
}

double Orbit::get_true_anomaly(double t)
{
	double mean_anomaly = mean_angular_motion_ * (t - epoch_) + mean_anomaly_at_epoch_;

	double eccentric_anomaly = mean_anomaly > 3.14159265358979323 ? mean_anomaly - eccentricity_ / 2.0 : mean_anomaly + eccentricity_ / 2.0;

	while (true)
	{
		double f = eccentric_anomaly - eccentricity_ * sin(eccentric_anomaly) - mean_anomaly;
		double df = 1.0 - eccentricity_ * cos(eccentric_anomaly);

		double delta = f / df;
		eccentric_anomaly -= delta;

		if (abs(delta) < 1e-8)
		{
			break;
		}
	}

	last_t_ = t;
	last_theta_ = 2.0 * atan(sqrt((1.0 + eccentricity_) / (1.0 - eccentricity_)) * tan(eccentric_anomaly / 2.0));

	return last_theta_;
}


Eigen::Vector3d Orbit::swap_yz(Eigen::Vector3d a)
{
	Eigen::Vector3d b;
	b << a(0), a(2), a(1);
	return b;
}