// class Equation
// This class is a wrapper for the Dormand-Prince method of solving
// ordinary differential equations.

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
		double reltol,
		double abstol,
		void* params = nullptr);

	// Create a copy of another Equation object
	Equation(const Equation& equation);

	// Assign state from an existing Equation object
	Equation& operator = (const Equation& equation);

	// Destroy Equation object
	~Equation();

	// Steps Equation until time is equal to tout
	// tout = final desired integration time
	void step(double tout);

	// Resets Equation to its initial values
	void reset();

	// Sets maximum number of iterations
	// n = maximum number of iterations
	void set_max_iter(int n);

	// Returns maximum number of iterations
	int get_max_iter() const;

	// Returns number of equations
	int get_neqn() const;

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

	// Returns current value of the derivative of parameter i
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

private:

	// Maximum number of iterations
	int max_iter_;

	bool reject_;

	// Number of equations in the problem to be solved
	int neqn_;

	// Relative error tolerance
	double reltol_;

	// Absolute error tolerance
	double abstol_;

	// Error from the previous step
	double errold_;

	// Step direction
	double d_;

	// Step size
	double h_;

	// Current time value
	double t_;
	
	// Initial time value
	double ti_;

	double tintrp_;

	// Current state of problem to be solved
	std::vector<double> y_;

	// Initial conditions of problem to be solved
	std::vector<double> yi_;

	// Current derivative of problem to be solved
	std::vector<double> yp_;

	// 
	std::vector<double> yw_;

	std::vector<double> yintrp_;

	std::vector<double> ypintrp_;

	// Coefficients for integration
	std::vector<double> k2_;
	std::vector<double> k3_;
	std::vector<double> k4_;
	std::vector<double> k5_;
	std::vector<double> k6_;
	std::vector<double> k7_;

	// Coefficients for polynomial interpolation
	std::vector<double> r1_;
	std::vector<double> r2_;
	std::vector<double> r3_;
	std::vector<double> r4_;
	std::vector<double> r5_;

	// Pointer to paramters required by derivative
	void* params_;

	// Pointer to function that calculates the derivative of the problem
	void(*f_)(double t, double y[], double yp[], void* params);

	// Dormand-Prince integrator
	void dopr5_();

	void set_initial_step_size_();

	void update_step_size_();

	void dense_();

	// Interpolates between the 
	// tout = final desired integration time
	void intrp_(double tout);
};