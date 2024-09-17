// Class Equation
// This class is an interface to several numeric integrators for solving
// ordinary differential equations.

#pragma once

#include <vector>
#include <string>

class Equation
{
public:

	// Create a default Equation object
	Equation();

	// Create an initialized Equation object
	// f = pointer to function that calculates the derivative of the problem
	// t = initial time
	// y = initial state of the problem
	// method = integrator method to be used
	// reltol = relative error tolerance
	// abstol = absolute error tolerance
	// params = optional pointer to parameters for f
	Equation(
		void f(double t, double y[], double yp[], void* params),
		double t,
		const std::vector<double>& y,
		std::string method = "DOPR5",
		double reltol = 1e-6,
		double abstol = 1e-6,
		void* params = nullptr);

	// Copy constructor
	Equation(const Equation& equation);

	// Assignment constructor
	Equation& operator = (const Equation& equation);

	// Destructor
	~Equation();

	// Steps Equation until time is equal to tout
	// tout = final desired integration time
	void step(double tout);

	// Returns status of the integrator
	int get_iflag() const;

	// Returns current time
	double get_t() const;

	// Returns current state at index i
	// i = index of desired value
	double get_y(int i) const;

	// Returns current state of the problem
	std::vector<double> get_y() const;

	// Copies current state into array
	// i = index of first value
	// n = number of values
	// x = pointer to array of length n where values will be stored
	void get_y(int i, int n, double* x) const;

	// Returns current value of derivative at index i
	// i = index of desired derivative
	double get_yp(int i) const;

	// Returns current derivative of the problem
	std::vector<double> get_yp() const;

	// Copies current derivatives into array
	// i = index of first derivative
	// n = number of derivatives
	// x = pointer to array of length n where derivatives will be stored
	void get_yp(int i, int n, double* x) const;

private:

	// Integrator method to be used
	enum class Method
	{
		NONE,
		DOPR5,
		DOPR853,
		ODE
	};

	Method method_;

	// Maximum number of iterations
	int max_iter_;

	// Flag which holds the return status of the integrator
	int iflag_;

	// Number of equations
	int neqn_;

	// Current relative error tolerance
	double reltol_;

	// Current absolute error tolerance
	double abstol_;

	// Current time value
	double t_;

	// Internal work space for the integrator
	std::vector<int> iwork_;

	// Internal work space for the integrator
	std::vector<double> work_;

	// Current state of problem
	std::vector<double> y_;

	// Current derivative of problem
	std::vector<double> yp_;

	// Pointer to paramters required by derivative
	void* params_;

	// Pointer to function that calculates the derivative of the problem
	void(*f_)(double t, double y[], double yp[], void* params);

	// Returns error string associated with the current value of iflag_
	std::string get_error_string() const;
};