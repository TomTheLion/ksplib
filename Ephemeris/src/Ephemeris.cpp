#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <string>
#include <vector>
#include <tuple>
#include "Ephemeris.h"

// Create an uninitialized Ephemeris object
Ephemeris::Ephemeris()
{

}

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
Ephemeris::Ephemeris(
	Equation& equation,
	std::vector<std::string> bodies,
	double muk,
	std::vector<double> mu,
	int time_steps,
	double delta_time,
	double maxerr,
	std::vector<double> info)
	: num_bodies_(equation.get_neqn() / 6), bodies_(bodies), muk_(muk), mu_(mu), time_steps_(time_steps),
	delta_time_(delta_time), max_time_(time_steps * delta_time), reference_time_(info[0]), time_scale_(info[1]),
	mu_scale_(info[2]), distance_scale_(info[3]), velocity_scale_(info[4])
{
	std::cout << "Constructing Ephemeris...\n";

	index_.resize(num_bodies_);
	num_coef_.resize(num_bodies_);

	index_[0] = 0;

	// set values of coefficient arrays which remain constant
	p_coef_[0] = 1.0;
	v_coef_[0] = 0.0;
	v_coef_[1] = 1.0;
	a_coef_[0] = 0.0;
	a_coef_[1] = 0.0;

	// matrix to store maximum number of Chebyshev polynomial values and their
	// derivatives at the beginning and end of 8 evenly spaced times within
	// each time step
	Eigen::MatrixXd tm18(18, 18);
	// matrix to store position and velocity weights
	Eigen::MatrixXd wm(18, 18);
	// matrix to store integrated position and velocity values
	Eigen::MatrixXd f(3 * num_bodies_, 18);

	// matrix to store Chebyshev polynomial values and their derivatives for
	// each body, stores values equal to the number of coefficients
	std::vector<Eigen::MatrixXd> tm(num_bodies_);
	// matrix c1 and c2 are used to simplify the form of the problem
	std::vector<Eigen::MatrixXd> c1(num_bodies_);
	std::vector<Eigen::MatrixXd> c2(num_bodies_);

	// start with the minimum number of 4 coefficients for each body
	for (int i = 0; i < num_bodies_; i++)
	{
		num_coef_[i] = 4;
	}

	// calculate the Chebyshev polynomial values and their derivatives
	for (int i = 0; i < 9; i++)
	{
		for (int j = 0; j < 18; j++)
		{
			double t = 1.0 - 0.25 * i;
			calculate_coef(18, t, true);
			tm18(2 * i, j) = p_coef_[j];
			tm18(2 * i + 1, j) = v_coef_[j];
		}
	}

	// calulate the values of the weight matrix
	wm.setIdentity();
	for (int i = 1; i < 18; i += 2)
	{
		wm(i, i) = 0.16;
	}

	// main solution loop to determine number of coefficients required for
	// each body to achieve desired error tolerance
	while (true)
	{
		bool fail_flag = false;

		std::cout << "Current coefficient array size: ";

		for (int i = 0; i < num_bodies_; i++)
		{
			std::cout << num_coef_[i] << ", ";
		}

		std::cout << "\n";

		// calculate starting index of each body in the ephemeris table
		for (int i = 1; i < num_bodies_; i++)
		{
			index_[i] = 3 * num_coef_[i - 1] * time_steps + index_[i - 1];
		}

		// calculate size of the ephemeris table
		size_ = 3 * num_coef_[num_bodies_ - 1] * time_steps + index_[num_bodies_ - 1];

		ephemeris_.resize(size_);

		// calculate c1 and c2 matricies for each body
		for (int i = 0; i < num_bodies_; i++)
		{
			int n = num_coef_[i];
			tm[i] = tm18.block(0, 0, 18, n);

			c1[i].resize(n + 4, n + 4);
			c1[i].setZero();
			c1[i].topLeftCorner(n, n) = tm[i].transpose() * wm * tm[i];
			c1[i].block(n + 0, 0, 1, n) = tm[i].row(0);
			c1[i].block(n + 1, 0, 1, n) = tm[i].row(1);
			c1[i].block(n + 2, 0, 1, n) = tm[i].row(16);
			c1[i].block(n + 3, 0, 1, n) = tm[i].row(17);
			c1[i].block(0, n, n, 4) = c1[i].block(n, 0, 4, n).transpose().eval();

			c2[i].resize(n + 4, 18);
			c2[i].setZero();
			c2[i].block(0, 0, n, 18) = tm[i].transpose() * wm;
			c2[i].block(n, 0, 2, 2) = Eigen::Matrix<double, 2, 2>::Identity();
			c2[i].bottomRightCorner(2, 2) = Eigen::Matrix<double, 2, 2>::Identity();
		}

		// calculate f vector for each body
		equation.reset();
		for (int time_step = 0; time_step < time_steps; time_step++)
		{
			for (int step = 0; step < 9; step++)
			{
				double tout = (step / 8.0 + time_step) *  delta_time;
				if (step > 0)
				{
					equation.step(tout);
				}
				int row = 0;
				for (int i = 0; i < 6 * num_bodies_; i++)
				{
					if (i % 6 < 3)
					{
						f(row, 2 * (8 - step)) = equation.get_y(i);
						row++;
						if (i % 6 == 2)
						{
							row -= 3;
						}
					}
					else
					{
						f(row, 2 * (8 - step) + 1) = equation.get_y(i) * 0.5 * delta_time;
						row++;
					}
				}
			}
			
			std::vector<bool> fail(num_bodies_, false);

			// calculate coefficient vector for each body, if error tolerances and the
			// maximum number of coefficients has not been reached, set fail to true and
			// increase the number of coefficients
			for (int i = 0; i < num_bodies_; i++)
			{
				for (int j = 0; j < 3; j++)
				{
					if (!fail[i])
					{

						Eigen::VectorXd temp = c1[i].fullPivHouseholderQr().solve(c2[i] * f.row(i * 3 + j).transpose()).topRows(num_coef_[i]).transpose();

						if (((tm[i] * temp).transpose() - f.row(i * 3 + j)).cwiseAbs().maxCoeff() > maxerr && num_coef_[i] < 18)
						{
							fail_flag = true;
							fail[i] = true;
							num_coef_[i]++;
							break;
						}

						for (int k = 0; k < num_coef_[i]; k++)
						{
							ephemeris_[index_[i] + 3 * num_coef_[i] * time_step + j * num_coef_[i] + k] = temp[k];
						}
					}
				}
			}

			// if any body fails, generate new ephemeris based on updated number of coefficients
			if (fail_flag)
			{
				break;
			}			
		}

		// if no bodies fail accept result and break out out loop
		if (!fail_flag)
		{
			break;
		}			
	}

	std::cout << "End of construction...\n\n";
}

 // Create an initialized Ephemeris object from a saved file
 // file_name = name of the file to load
Ephemeris::Ephemeris(std::string file_name)
{
	// set values of coefficient arrays which remain constant
	p_coef_[0] = 1.0;
	v_coef_[0] = 0.0;
	v_coef_[1] = 1.0;
	a_coef_[0] = 0.0;
	a_coef_[1] = 0.0;

	// attempt to load ephemeris file
	std::ifstream header_in(file_name + ".cfg");
	std::ifstream binary_in(file_name + ".bin", std::ios::in | std::ios::binary);

	if (!header_in || !binary_in)
	{
		throw std::runtime_error("Runtime Error: Ephemeris file not found");
	}

	std::string header_in_line;
	std::vector<std::string> header_in_vector;
	while (std::getline(header_in, header_in_line))
	{
		header_in_vector.push_back(header_in_line.substr(header_in_line.find(": ") + 2));
	}

	size_ = std::stoi(header_in_vector[0]);
	num_bodies_ = std::stoi(header_in_vector[1]);
	time_steps_ = std::stoi(header_in_vector[2]);
	delta_time_ = std::stod(header_in_vector[3]);
	max_time_ = time_steps_ * delta_time_;
	muk_ = std::stod(header_in_vector[4]);
	reference_time_ = std::stod(header_in_vector[5]);
	time_scale_ = std::stod(header_in_vector[6]);
	mu_scale_ = std::stod(header_in_vector[7]);
	distance_scale_ = std::stod(header_in_vector[8]);
	velocity_scale_ = std::stod(header_in_vector[9]);

	auto str_to_str_vector = [](std::string str, std::vector<std::string>& vec)
	{
		vec.clear();
		std::stringstream ss(str);
		std::string substr;
		while (std::getline(ss, substr, ',')) {
			substr.erase(remove(substr.begin(), substr.end(), ' '), substr.end());
			vec.push_back(substr);
		}
	};

	auto str_to_int_vector = [](std::string str, std::vector<int>& vec)
	{
		vec.clear();
		std::stringstream ss(str);
		std::string substr;
		while (std::getline(ss, substr, ',')) {
			vec.push_back(std::stoi(substr));
		}
	};

	auto str_to_double_vector = [](std::string str, std::vector<double>& vec)
	{
		vec.clear();
		std::stringstream ss(str);
		std::string substr;
		while (std::getline(ss, substr, ',')) {
			vec.push_back(std::stod(substr));
		}
	};

	str_to_int_vector(header_in_vector[10], index_);
	str_to_int_vector(header_in_vector[11], num_coef_);
	str_to_str_vector(header_in_vector[12], bodies_);
	str_to_double_vector(header_in_vector[13], mu_);

	ephemeris_.resize(size_);

	for (int i = 0; i < size_; i++)
	{
		binary_in.read((char *)&ephemeris_[i], sizeof ephemeris_[i]);
	}
}

// Destory Ephemeris object
Ephemeris::~Ephemeris()
{

}

 // Save an Ephemeris to file
 // file_name = name of the file to save
void Ephemeris::save(std::string file_name)
{
	// attempt to save ephemeris file
	std::ofstream header_out(file_name + ".cfg");
	std::ofstream binary_out(file_name + ".bin", std::ios::out | std::ios::binary);

	if (!header_out) {
		std::cout << "Cannot open file: " << file_name + ".cfg" << std::endl;
		return;
	}

	if (!binary_out) {
		std::cout << "Cannot open file: " << file_name + ".bin" << std::endl;
		return;
	}

	header_out << std::setprecision(17);
	header_out << "size: " << size_ << '\n';
	header_out << "num bodies: " << num_bodies_ << '\n';
	header_out << "time steps: " << time_steps_ << '\n';
	header_out << "delta time: " << delta_time_ << '\n';
	header_out << "mu primary: " << muk_ << '\n';
	header_out << "reference time: " << reference_time_ << '\n';
	header_out << "time scale: " << time_scale_ << '\n';
	header_out << "mu scale: " << mu_scale_ << '\n';
	header_out << "distance scale: " << distance_scale_ << '\n';
	header_out << "velocity scale: " << velocity_scale_ << '\n';

	header_out << "index: ";
	for (int i = 0; i < num_bodies_; i++)
	{
		header_out  << index_[i];
		if (i < num_bodies_ - 1)
		{
			header_out << ", ";
		}
	}

	header_out << "\nnum coef: ";
	for (int i = 0; i < num_bodies_; i++)
	{
		header_out << num_coef_[i];
		if (i < num_bodies_ - 1)
		{
			header_out << ", ";
		}
	}

	header_out << "\nbodies: ";
	for (int i = 0; i < num_bodies_; i++)
	{
		header_out << bodies_[i];
		if (i < num_bodies_ - 1)
		{
			header_out << ", ";
		}
	}

	header_out << "\nmu: ";
	for (int i = 0; i < num_bodies_; i++)
	{
		header_out << mu_[i];
		if (i < num_bodies_ - 1)
		{
			header_out << ", ";
		}
	}

	header_out.close();

	for (int i = 0; i < size_; i++)
	{
		binary_out.write((char *)&ephemeris_[i], sizeof ephemeris_[i]);
	}

	binary_out.close();
}

// Test Ephemeris against an Equation, provides maximum position and
// velocity error values for the time span of the ephemeris
// equation = Equation to test against the ephemeris
void Ephemeris::test(Equation& equation)
{
	std::vector<double> max_poserr(num_bodies_, 0.0);
	std::vector<double> max_velerr(num_bodies_, 0.0);
	Eigen::Vector3d integrated_position, integrated_velocity;

	// calculate position and velocity from Equation and Ephemeris and save the
	// magnitude of their difference if it is higher than the current saved
	// magnitude for each body.
	equation.reset();
	for (int step = 0; step < 16 * time_steps_; step++)
	{
		double tout = step / 16.0 * delta_time_;
		if (step > 0)
		{
			equation.step(tout);
		}

		for (int i = 0; i < num_bodies_; i++)
		{
			integrated_position << equation.get_y(6 * i + 0), equation.get_y(6 * i + 1), equation.get_y(6 * i + 2);
			integrated_velocity << equation.get_y(6 * i + 3), equation.get_y(6 * i + 4), equation.get_y(6 * i + 5);

			double poserror = (get_position(i, equation.get_t()) - integrated_position).norm();
			double velerror = (get_velocity(i, equation.get_t()) - integrated_velocity).norm();

			if (abs(poserror) * get_distance_scale() > max_poserr[i])
			{
				max_poserr[i] = abs(poserror) * get_distance_scale();
			}				

			if (abs(velerror) * get_velocity_scale() > max_velerr[i])
			{
				max_velerr[i] = abs(velerror) * get_velocity_scale();
			}				
		}
	}

	std::cout << std::setw(2) << " - : Test Maximum Error" << std::endl;
	std::cout << std::setw(2) << " - : Position : Velocity" << std::endl;

	for (int i = 0; i < num_bodies_; i++)
	{
		std::cout << std::setw(8) << std::left << bodies_[i] << std::right << std::scientific << std::setprecision(4) << ": " << max_poserr[i] << " : " << max_velerr[i] << std::endl;
	}
}

// Calculates position, velocity, and acceleration coefficients and stores
// them in p_coef_, v_coef_, and a_coef
// n = index of desired body
// t = desired time
// vflag = flag to indicate if velocity and/or acceleration coefficients
// are to be calculated. 0 = only position, 1 = only position and velocity,
// 2 = position, velocity, and acceleration
void Ephemeris::calculate_coef(const int n, const double t, const int vflag)
{
	double twot = t + t;
	p_coef_[1] = t;
	double* p_ptr = &p_coef_[2];
	
	for (int i = n - 2; i; i--, p_ptr++)
	{
		*p_ptr = twot * p_ptr[-1] - p_ptr[-2];
	}
		
	if (vflag == 0) { return; }

	double* v_ptr = &v_coef_[2];
	p_ptr = &p_coef_[1];

	for (int i = n - 2; i; i--, v_ptr++, p_ptr++)
	{
		*v_ptr = twot * v_ptr[-1] + *p_ptr + *p_ptr - v_ptr[-2];
	}
		
	if (vflag == 1) { return; }

	double* a_ptr = &a_coef_[2];
	v_ptr = &v_coef_[1];

	for (int i = n - 2; i; i--, a_ptr++, v_ptr++)
	{
		*a_ptr = twot * a_ptr[-1] + 4.0 * *v_ptr - a_ptr[-2];
	}	
}

// Returns position of a body at a specified time by index
// b = index of the desired body
// t = desired time
Eigen::Vector3d Ephemeris::get_position(const int b, const double t)
{
	double step;
	double tn = 2.0 * modf(t / delta_time_, &step) - 1.0;
	calculate_coef(num_coef_[b], tn, 0);
	double* c_ptr = &ephemeris_[0] + index_[b] + 3 * int(step) * num_coef_[b];

	Eigen::Vector3d position{ 0, 0, 0 };

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < num_coef_[b]; j++, c_ptr++)
		{
			position(i) += *c_ptr * p_coef_[j];
		}
	}

	return position;
}

// Returns position of a body at a specified time by name
// body = name of the desired body
// t = desired time
Eigen::Vector3d Ephemeris::get_position(const std::string body, const double t)
{
	int b = get_index_of_body(body);

	return get_position(b, t);
}

// Returns velocity of a body at a specified time by index
// b = index of the desired body
// t = desired time
Eigen::Vector3d Ephemeris::get_velocity(const int b, const double t)
{
	double step;
	double tn = 2.0 * modf(t / delta_time_, &step) - 1.0;
	calculate_coef(num_coef_[b], tn, 1);
	double* c_ptr = &ephemeris_[0] + index_[b] + 3 * int(step) * num_coef_[b];

	Eigen::Vector3d velocity{ 0, 0, 0 };

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < num_coef_[b]; j++, c_ptr++)
		{
			velocity(i) += *c_ptr * v_coef_[j];
		}
	}

	return velocity * 2.0 / delta_time_;
}

// Returns velocity of a body at a specified time by name
// body = name of the desired body
// t = desired time
Eigen::Vector3d Ephemeris::get_velocity(const std::string body, const double t)
{
	int b = get_index_of_body(body);

	return get_velocity(b, t);
}

// Returns acceleration of a body at a specified time by index
// b = index of the desired body
// t = desired time
Eigen::Vector3d Ephemeris::get_acceleration(const int b, const double t)
{
	double step;
	double tn = 2.0 * modf(t / delta_time_, &step) - 1.0;
	calculate_coef(num_coef_[b], tn, 2);
	double* c_ptr = &ephemeris_[0] + index_[b] + 3 * int(step) * num_coef_[b];

	Eigen::Vector3d acceleration{ 0, 0, 0 };

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < num_coef_[b]; j++, c_ptr++)
		{
			acceleration(i) += *c_ptr * a_coef_[j];
		}
	}

	return acceleration * 4.0 / delta_time_ / delta_time_;
}

// Returns acceleration of a body at a specified time by name
// body = name of the desired body
// t = desired time
Eigen::Vector3d Ephemeris::get_acceleration(const std::string body, const double t)
{
	int b = get_index_of_body(body);

	return get_acceleration(b, t);
}

// Returns position and velocity values of a body at a specified time into
// as Eigen vectors
// b = index of the desired body
// t = desired time

std::tuple<Eigen::Vector3d, Eigen::Vector3d> Ephemeris::get_position_velocity(const int b, const double t)
{
	double step;
	double tn = 2.0 * modf(t / delta_time_, &step) - 1.0;
	calculate_coef(num_coef_[b], tn, 1);
	double* c_ptr = &ephemeris_[0] + index_[b] + 3 * int(step) * num_coef_[b];

	Eigen::Vector3d position { 0.0, 0.0, 0.0 };
	Eigen::Vector3d velocity { 0.0, 0.0, 0.0 };

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < num_coef_[b]; j++, c_ptr++)
		{
			position(i) += *c_ptr * p_coef_[j];
			velocity(i) += *c_ptr * v_coef_[j];
		}
	}

	velocity *= 2.0 / delta_time_;

	return { position, velocity };
}

// Returns position and velocity values of a body at a specified time into
// as Eigen vectors
// body = name of the desired body
// t = desired time
std::tuple<Eigen::Vector3d, Eigen::Vector3d> Ephemeris::get_position_velocity(const std::string body, const double t)
{
	int b = get_index_of_body(body);

	return get_position_velocity(b, t);
}

// Returns position, velocity, and acceleration values of a body at a
// specified time as three Eigen vectors
// b = index of the desired body
// t = desired time
std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d> Ephemeris::get_position_velocity_acceleration(const int b, const double t)
{
	double step;
	double tn = 2.0 * modf(t / delta_time_, &step) - 1.0;
	calculate_coef(num_coef_[b], tn, 2);
	double* c_ptr = &ephemeris_[0] + index_[b] + 3 * int(step) * num_coef_[b];

	Eigen::Vector3d position { 0.0, 0.0, 0.0 };
	Eigen::Vector3d velocity { 0.0, 0.0, 0.0 };
	Eigen::Vector3d acceleration { 0.0, 0.0, 0.0 };

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < num_coef_[b]; j++, c_ptr++)
		{
			position(i) += *c_ptr * p_coef_[j];
			velocity(i) += *c_ptr * v_coef_[j];
			acceleration(i) += *c_ptr * a_coef_[j];
		}
	}

	velocity *= 2.0 / delta_time_;
	acceleration *= 4.0 / delta_time_ / delta_time_;

	return { position, velocity, acceleration };
}

// Returns position, velocity, and acceleration values of a body at a
// specified time as three Eigen vectors
// body = name of the desired body
// t = desired time
std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Vector3d> Ephemeris::get_position_velocity_acceleration(const std::string body, const double t)
{
	int b = get_index_of_body(body);

	return get_position_velocity_acceleration(b, t);
}

int Ephemeris::get_num_bodies() const
{
	return num_bodies_;
}

double Ephemeris::get_max_time() const
{
	return max_time_;
}

int Ephemeris::get_index_of_body(std::string body) const
{
	return std::distance(bodies_.begin(), std::find(bodies_.begin(), bodies_.end(), body));
}

std::vector<std::string> Ephemeris::get_bodies() const
{
	return bodies_;
}

double Ephemeris::get_mu(const int b) const
{
	return mu_[b];
}

double Ephemeris::get_mu(const std::string body) const
{
	int b = get_index_of_body(body);

	return mu_[b];
}

double Ephemeris::get_muk() const
{
	return muk_;
}

std::vector<double> Ephemeris::get_mu() const
{
	return mu_;
}

double Ephemeris::get_reference_time() const
{
	return reference_time_;
}

double Ephemeris::get_time_scale() const
{
	return time_scale_;
}

double Ephemeris::get_ephemeris_time(Jdate jd) const
{
	return (jd.get_kerbal_time() - get_reference_time()) / get_time_scale();	
}

Jdate Ephemeris::get_jdate(const double t) const
{
	Jdate jd;
	jd.set_kerbal_time(t * get_time_scale() + get_reference_time());
	return jd;
}

double Ephemeris::get_mu_scale() const
{
	return mu_scale_;
}

double Ephemeris::get_distance_scale() const
{
	return distance_scale_;
}

double Ephemeris::get_velocity_scale() const
{
	return velocity_scale_;
}