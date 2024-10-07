// class Equation
// This class is a wrapper for the integrator ODE. It handles the allocation
// storage, and deletion of memory required by ODE in addition to simplifying
// the calling procedure.

#pragma once

#include <vector>

class Equation
{
public:

	// Create an uninitialized Equation object
	Equation();

	// Create an initialized Equation object
	// f = pointer to function that calculates the derivative of the problem
	// neqn = number of equations in the problem to be solved
	// y = pointer to array of length neqn where the initial values of the
	// problem are stored
	// t = initial time
	// relerr = relative error tolerance
	// abserr = absolution error tolerance
	Equation(
		void f(double t, double y[], double yp[], void* params),
		int neqn,
		const double* y,
		double t,
		double relerr,
		double abserr,
		void* params=NULL);

	// Create a copy of another Equation object
	Equation(const Equation& equation);

	// Assign state from an existing Equation object
	Equation& operator = (const Equation& equation);

	// Destroy Equation object
	~Equation();

	// Maximum number of outerloop iterations of ODE, total number of ODE
	// iterations is equal to 500 * max_iter
	unsigned int max_iter;

	// Steps Equation until time is equal to tout
	// tout = final desired integration time
	void step(double tout);

	// Steps Equation until time is equal to tout
	// Prevents ODE from calling f where t > tout
	// tout = final desired integration time
	void stepn(double tout);

	// Resets Equation to its initial values
	void reset();

	// Reinitializes Equation by setting iflag to 1
	void reinit();

	// Returns iflag
	int get_iflag() const;

	// Returns number of equations
	int get_neqn() const;

	// Returns current relative error
	double get_relerr() const;

	// Returns current absolute error
	double get_abserr() const;

	// Returns current time
	double get_t() const;

	// Returns initial time
	double get_ti() const;

	// Returns current value of parameter i
	// i = index of desired parameter
	double get_y(int i) const;

	// Returns initial value of parameter i
	// i = index of desired parameter
	double get_yi(int i) const;

	// Returns current value of derivative of parameter i
	// i = index of desired parameter
	double get_yp(int i) const;

	// Copies current values of parameters into array
	// i = index of first parameter
	// n = number of parameters
	// x = pointer to array of length n where values will be stored
	void get_y(int i, int n, double* x) const;

	// Copies initial values of parameters into array
	// i = index of first parameter
	// n = number of parameters
	// x = pointer to array of length n where values will be stored
	void get_yi(int i, int n, double* x) const;

	// Copies derivatives of parameters into array
	// i = index of first parameter
	// n = number of parameters
	// x = pointer to array of length n where values will be stored
	void get_yp(int i, int n, double* x) const;

	// Sets initial time
	// Following change to intial time and/or initial conditions reset
	// should be called
	// x = desired initial time
	void set_ti(double x);

	// Sets initial condition
	// Following change to intial time and/or initial conditions reset
	// should be called
	// i = index of desired parameter to change
	// x = value of desired parameter
	void set_yi(int i, double x);

	// Copies values from array into initial conditions
	// Following change to intial time and/or initial conditions reset
	// should be called
	// i = index of first parameter
	// n = number of parameters
	// x = pointer to array of length n with desired values
	void set_yi(int i, int n, const double* x);

	// Sets param pointer
	void set_params(void* p);

private:

	// Flag which holds the return status of ODE
	int iflag_;

	// Internal work space for ODE
	int iwork_[5];

	// Number of equations in the problem to be solved
	int neqn_;

	// Current relative error tolerance
	double relerr_;

	// Initial relative error tolerance
	double relerri_;

	// Current absolute error tolerance
	double abserr_;

	// Initial absolute error tolerance
	double abserri_;

	// Current time value
	double t_;

	// Initial time value
	double ti_;

	// Internal work space for ODE
	std::vector<double> work_;

	// Current state of problem to be solved
	std::vector<double> y_;

	// Initial conditions of problem to be solved
	std::vector<double> yi_;

	// Pointer to paramters required by derivative
	void* params_;

	// Returns error string associated with the current value of iflag_
	std::string get_error_string(int iflag) const;

	// Pointer to function that calculates the derivative of the problem
	void(*f_)(double t, double y[], double yp[], void* params);
};
