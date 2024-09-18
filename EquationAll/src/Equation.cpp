#include <iostream>
#include "Equation.h"
#include "adb2.h"
#include "bosh3.h"
#include "dopr5.h"

// Create a default Equation object
Equation::Equation()
{
	method_ = Method::NONE;
	max_iter_ = 0;
	iflag_ = 0;
	neqn_ = 0;
	reltol_ = 0.0;
	abstol_ = 0.0;
	t_ = 0.0;
	params_ = nullptr;
	f_ = nullptr;
}

// Create an initialized Equation object
// f = pointer to function that calculates the derivative of the problem
// t = initial time
// y = initial state of the problem
// method = integrator method to be used
// reltol = relative error tolerance
// abstol = absolute error tolerance
// params = optional pointer to parameters for f
Equation::Equation(
	void f(double t, double y[], double yp[], void* params),
	double t,
	const std::vector<double>& y,
	std::string method,
	double reltol,
	double abstol,
	void* params)
	: f_(f), t_(t), y_(y), reltol_(reltol), abstol_(abstol), params_(params)
{
	method_ = Method::NONE;
	max_iter_ = 0;
	iflag_ = 0;
	neqn_ = y.size();

	if (method == "ADB2")
	{
		method_ = Method::ADB2;
		max_iter_ = 100000;
		adb2::init(f_, iflag_, neqn_, t_, y_, yp_, iwork_, work_, params_);
	}
	else if (method == "BOSH3")
	{
		method_ = Method::BOSH3;
		max_iter_ = 100000;
		bosh3::init(f_, iflag_, neqn_, t_, y_, yp_, iwork_, work_, params_);
	}
	else if (method == "DOPR5")
	{
		method_ = Method::DOPR5;
		max_iter_ = 100000;
		dopr5::init(f_, iflag_, neqn_, t_, y_, yp_, iwork_, work_, params_);
	}
	else if (method == "DOPR853")
	{

	}
	else if (method == "ODE")
	{

	}
	else
	{
		throw std::runtime_error("Equation failed, unknown method.");
	}
}

// Copy constructor
Equation::Equation(const Equation& equation)
{
	method_ = equation.method_;
	max_iter_ = equation.max_iter_;
	iflag_ = equation.iflag_;
	neqn_ = equation.neqn_;
	reltol_ = equation.reltol_;
	abstol_ = equation.abstol_;
	t_ = equation.t_;
	iwork_ = equation.iwork_;
	work_ = equation.work_;
	y_ = equation.y_;
	yp_ = equation.yp_;
	params_ = equation.params_;
	f_ = equation.f_;
}

// Assignment constructor
Equation& Equation::operator = (const Equation& equation)
{
	if (&equation == this)
	{
		return *this;
	}

	method_ = equation.method_;
	max_iter_ = equation.max_iter_;
	iflag_ = equation.iflag_;
	neqn_ = equation.neqn_;
	reltol_ = equation.reltol_;
	abstol_ = equation.abstol_;
	t_ = equation.t_;
	iwork_ = equation.iwork_;
	work_ = equation.work_;
	y_ = equation.y_;
	yp_ = equation.yp_;
	params_ = equation.params_;
	f_ = equation.f_;

	return *this;
}

// Destructor
Equation::~Equation()
{

}

// Steps Equation until time is equal to tout
// tout = final desired integration time
void Equation::step(double tout)
{
	switch (method_)
	{
	case Method::ADB2:
		adb2::step(f_, max_iter_, iflag_, neqn_, reltol_, abstol_, t_, tout, y_, yp_, iwork_, work_, params_);
		return;
	case Method::BOSH3:
		bosh3::step(f_, max_iter_, iflag_, neqn_, reltol_, abstol_, t_, tout, y_, yp_, iwork_, work_, params_);
		return;
	case Method::DOPR5:
		dopr5::step(f_, max_iter_, iflag_, neqn_, reltol_, abstol_, t_, tout, y_, yp_, iwork_, work_, params_);
		return;
	case Method::DOPR853:
		return;
	case Method::ODE:
		return;
	default:
		return;
	}
}

// Returns status of the integrator
int Equation::get_iflag() const
{
	return iflag_;
}

// Returns current time
double Equation::get_t() const
{
	return t_;
}

// Returns current state at index i
// i = index of desired value
double Equation::get_y(int i) const
{
	return y_[i];
}

// Returns current state of the problem
std::vector<double> Equation::get_y() const
{
	return y_;
}

// Copies current state into array
// i = index of first value
// n = number of values
// x = pointer to array of length n where values will be stored
void Equation::get_y(int i, int n, double* x) const
{
	const double* a = &y_[i];
	for (int j = 0; j < n; j++)
		*x++ = *a++;
}

// Returns current value of derivative at index i
// i = index of desired derivative
double Equation::get_yp(int i) const
{
	return yp_[i];
}

// Returns current derivative of the problem
std::vector<double> Equation::get_yp() const
{
	return yp_;
}

// Copies current derivatives into array
// i = index of first derivative
// n = number of derivatives
// x = pointer to array of length n where derivatives will be stored
void Equation::get_yp(int i, int n, double* x) const
{
	const double* a = &yp_[i];
	for (int j = 0; j < n; j++)
		*x++ = *a++;
}

std::string Equation::get_error_string() const
{
	switch (method_)
	{
	case Method::ADB2:
		return "";
	case Method::BOSH3:
		return "";
	case Method::DOPR5:
		return "";
	case Method::DOPR853:
		return "";
	case Method::ODE:
		return "";
	default:
		return "";
	}
}