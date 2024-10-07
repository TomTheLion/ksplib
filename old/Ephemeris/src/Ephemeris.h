// class Ephemeris
// This class creates a planetary ephemeris from an Equation object or from a
// saved ephemeris file. It can then return values of positions, velocities,
// or accelerations of any body within the time frame of the ephemeris.

#pragma once

#include <Eigen/Dense>
#include "Equation.h"
#include "Jdate.h"

class Ephemeris
{
public:

	// Create an uninitialized Ephemeris object
	Ephemeris();

	// Create an initialized Ephemeris object
	// equation = Equation object initialized with the initial conditions of
	// the planetary system and the n body derivative function
	// bodies = vector of the names of all bodies, excluding the central body
	// muk = gravitational parameter of the central body
	// mu = vector of gravitational parameters of all bodies, excluding the
	// central body
	// time_steps = number of time steps in the ephemeris 
	// delta_time = time interval of each time step
	// maxerr = maximum desired error at each test point
	// info = vector storing reference time, time scale, gravitational
	// parameter scale, distance scale, and velocity scale
	Ephemeris(
		Equation& equation,
		std::vector<std::string> bodies,
		double muk,
		std::vector<double> mu,
		int time_steps,
		double delta_time,
		double maxerr,
		std::vector<double> info);

	// Create an initialized Ephemeris object from a saved file
	// file_name = name of the file to load
	Ephemeris(std::string file_name);

	// Destory Ephemeris object
	~Ephemeris();

	// Save an Ephemeris to file
	// file_name = name of the file to save
	void save(std::string file_name);

	// Test Ephemeris against an Equation, provides maximum position and
	// velocity error values for the time span of the ephemeris
	// equation = Equation to test against the ephemeris
	void test(Equation& equation);

	// Returns position of a body at a specified time by index
	// b = index of the desired body
	// t = desired time
	Eigen::Vector3d get_position(const int b, const double t);

	// Returns position of a body at a specified time by name
	// body = name of the desired body
	// t = desired time
	Eigen::Vector3d get_position(const std::string body, const double t);

	// Returns velocity of a body at a specified time by index
	// b = index of the desired body
	// t = desired time
	Eigen::Vector3d get_velocity(const int b, const double t);

	// Returns velocity of a body at a specified time by name
	// body = name of the desired body
	// t = desired time
	Eigen::Vector3d get_velocity(const std::string body, const double t);
	
	// Returns acceleration of a body at a specified time by index
	// b = index of the desired body
	// t = desired time
	Eigen::Vector3d get_acceleration(const int b, const double t);

	// Returns acceleration of a body at a specified time by name
	// body = name of the desired body
	// t = desired time
	Eigen::Vector3d get_acceleration(const std::string body, const double t);

	// Returns position and velocity values of a body at a specified time into
	// as Eigen vectors
	// b = index of the desired body
	// t = desired time
	std::tuple<Eigen::Vector3d, Eigen::Vector3d> get_position_velocity(const int b, const double t);

	// Returns position and velocity values of a body at a specified time into
	// as Eigen vectors
	// body = name of the desired body
	// t = desired time
	std::tuple<Eigen::Vector3d, Eigen::Vector3d> get_position_velocity(const std::string body, const double t);

	// Returns position, velocity, and acceleration values of a body at a
	// specified time as three Eigen vectors
	// b = index of the desired body
	// t = desired time
	std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d> get_position_velocity_acceleration(const int b, const double t);

	// Returns position, velocity, and acceleration values of a body at a
	// specified time as three Eigen vectors
	// body = name of the desired body
	// t = desired time
	std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d> get_position_velocity_acceleration(const std::string body, const double t);

	// Returns number of bodies in the ephemeris, not including the central body
	int get_num_bodies() const;

	// Returns the maximum time value of the ephemeris
	double get_max_time() const;

	// Returns the index of a body by name
	// body = name of the desired body
	int get_index_of_body(std::string body) const;

	// Returns a vector containing names of all bodies, excluding the central body
	std::vector<std::string> get_bodies() const;

	// Returns the gravitaional parameter of a body by index
	// b = index of desired body
	double get_mu(const int b) const;

	// Returns the gravitaional parameter of a body by name
	// body = name of the desired body
	double get_mu(const std::string body) const;

	// Returns the gravitational parameter of the central body
	double get_muk() const;

	// Returns a vector containing gravitional parameter values of all bodies,
	// excluding the central body
	std::vector<double> get_mu() const;

	// Returns the reference time of the ephemeris
	double get_reference_time() const;

	// Returns the time scale of the ephemeris
	double get_time_scale() const;

	// Returns ephemeris time from Julain date
	double get_ephemeris_time(Jdate jd) const;

	// Returns Julian date from ephemeris time
	Jdate get_jdate(const double t) const;

	// Returns the gravitational parameter scale of the ephemeris
	double get_mu_scale() const;

	// Returns the distance scale of the ephemeris
	double get_distance_scale() const;
	
	// Returns the velocity scale of the ephemeris
	double get_velocity_scale() const;

private:

	// Number of bodies in the ephemeris, excluding the central body
	int num_bodies_;

	// Number of time steps in the ephemeris
	int time_steps_;

	// Time interval of each time step
	double delta_time_;

	// Maxmimum time value of the ephemeris, equal to time_steps_ * delta_time_
	double max_time_;

	// Gravitational parameter of central body
	double muk_;

	// Names of all bodies in the ephemeris, excluding the central body
	std::vector<std::string> bodies_;

	// Gravitational parameters of all bodies, excluding the central body
	std::vector<double> mu_;

	// Reference time of the ephemeris (not scaled)
	double reference_time_;

	// Time scale of the ephemeris (time)
	double time_scale_;

	// Gravitational parameter scale of the ephemeris (length^3 / time^2)
	double mu_scale_;

	// Distance scale of the ephemeris (length)
	double distance_scale_;

	// Velocity scale of the ephemeris (length / time)
	double velocity_scale_;

	// Size of the ephemeris table, equal to sum of 3 * num_ceof_ * time_steps_
	// over every body excluding the central body 
	int size_;

	// Vector of the starting index for each body in the ephemeris table
	std::vector<int> index_;

	// Vector of the number of coefficients stored for each body
	std::vector<int> num_coef_;

	// Ephemeris table that holds the coefficients for all bodies, excluding 
	// the central body
	std::vector<double> ephemeris_;

	// Internal workspace which holds calculated position coefficients
	double p_coef_[18];

	// Internal workspace which holds calculated velocity coefficients
	double v_coef_[18];

	// Internal workspace which holds calculated acceleration coefficients
	double a_coef_[18];

	// Calculates position, velocity, and acceleration coefficients and stores
	// them in p_coef_, v_coef_, and a_coef
	// n = index of desired body
	// t = desired time
	// vflag = flag to indicate if velocity and/or acceleration coefficients
	// are to be calculated. 0 = only position, 1 = only position and velocity,
	// 2 = position, velocity, and acceleration
	void calculate_coef(const int n, const double t, const int vflag);
};

