#include <iostream>
#include <string>
#include "Equation.h"
#include "ode.hpp"

// Create an uninitialized Equation object
Equation::Equation()
{

}

// Create an initialized Equation object
// f = pointer to function that calculates the derivative of the problem
// neqn = number of equations in the problem to be solved
// y = pointer to array of length neqn where the initial values of the
// problem are stored
// t = initial time
// relerr = relative error tolerance
// abserr = absolution error tolerance
Equation::Equation(
	void f(double t, double y[], double yp[], void* params),
	int neqn,
	const double* y,
	double t,
	double relerr,
	double abserr,
	void* params)
	: max_iter(500), iflag_(1), neqn_(neqn), abserr_(abserr), relerr_(relerr), t_(t), ti_(t), f_(f), params_(params)
{
	abserri_ = abserr;
	relerri_ = relerr;
	y_.resize(neqn_);
	yi_.resize(neqn_);
	work_.resize(100 + 21 * neqn_);

	for (int i = 0; i < neqn_; i++)
	{
		y_[i] = y[i];
		yi_[i] = y[i];
	}
}

// Create a copy of another Equation object
Equation::Equation(const Equation& equation)
	: max_iter(equation.max_iter), iflag_(equation.iflag_), neqn_(equation.neqn_), abserr_(equation.abserr_),
	relerr_(equation.relerr_), t_(equation.t_), ti_(equation.ti_), f_(equation.f_), params_(equation.params_)
{
	abserri_ = equation.abserr_;
	relerri_ = equation.relerr_;
	y_.resize(neqn_);
	yi_.resize(neqn_);
	work_.resize(100 + 21 * neqn_);

	for (int i = 0; i < neqn_; i++)
	{
		y_[i] = equation.y_[i];
		yi_[i] = equation.yi_[i];
	}

	for (int i = 0; i < 5; i++)
	{
		iwork_[i] = equation.iwork_[i];
	}

	for (int i = 0; i < 100 + 21 * neqn_; i++)
	{
		work_[i] = equation.work_[i];
	}
}

// Assign state from an existing Equation object
Equation& Equation::operator = (const Equation& equation)
{
	if (&equation == this)
	{
		return *this;
	}

	max_iter = equation.max_iter;
	iflag_ = equation.iflag_;
	neqn_ = equation.neqn_;
	abserr_ = equation.abserr_;
	abserri_ = equation.abserr_;
	relerr_ = equation.relerr_;
	relerri_ = equation.relerr_;
	t_ = equation.t_;
	ti_ = equation.ti_;
	f_ = equation.f_;
	params_ = equation.params_;

	y_.resize(neqn_);
	yi_.resize(neqn_);
	work_.resize(100 + 21 * neqn_);

	for (int i = 0; i < neqn_; i++)
	{
		y_[i] = equation.y_[i];
		yi_[i] = equation.yi_[i];
	}

	for (int i = 0; i < 5; i++)
	{
		iwork_[i] = equation.iwork_[i];
	}

	for (int i = 0; i < 100 + 21 * neqn_; i++)
	{
		work_[i] = equation.work_[i];
	}

	return *this;
}

// Destroy equation object
Equation::~Equation()
{

}

// Steps Equation until time is equal to tout
// tout = final desired integration time
void Equation::step(double tout)
{
	int iter = 0;
	bool tflag = false;

	// if tout is equal to the current time do nothing
	if (t_ != tout)
	{
		while (iter < max_iter)
		{
			iter++;

			// call to ode to step to tout
			ode(f_, neqn_, &y_[0], t_, tout, relerr_, abserr_, iflag_, &work_[0], iwork_, params_);

			// break loop if ode reaches tout, if ode fails to reach tout due to too small
			// tolerances, or the maximum number of iterations have been reached
			if (!(abs(iflag_) > 2 && iter < max_iter))
			{
				break;
			}

			// if tolerances where changed set tflag to true
			if (abs(iflag_ == 3))
			{
				tflag = true;
			}
		}

		if (tflag) {
			std::cerr << "Equation failed, " + get_error_string(3);
		}

		// if the step failed throw with error message
		if (iflag_ != 2)
		{
			throw std::runtime_error("Equation failed, " + get_error_string(iflag_));
		}
	}
}

// Steps Equation until time is equal to tout
// Prevents ODE from calling f where t > tout
// tout = final desired integration time
void Equation::stepn(double tout)
{
	// setting iflag_ to be negative indicates to ODE that the f_ cannot be
	// evaluated past tout.
	iflag_ = -abs(iflag_);

	if (t_ != tout)
	{
		step(tout);
	}
}

// Resets Equation to its initial values
void Equation::reset()
{
	iflag_ = 1;
	t_ = ti_;
	abserr_ = abserri_;
	relerr_ = relerri_;
	for (int i = 0; i < neqn_; i++)
	{
		y_[i] = yi_[i];
	}
}

// Reinitializes Equation by setting iflag to 1
void Equation::reinit()
{
	iflag_ = 1;
}

int Equation::get_iflag() const
{
	return iflag_;
}

int Equation::get_neqn() const
{
	return neqn_;
}

double Equation::get_relerr() const
{
	return relerr_;
}

double Equation::get_abserr() const
{
	return abserr_;
}

double Equation::get_t() const
{
	return t_;
}

double Equation::get_ti() const
{
	return ti_;
}

double Equation::get_y(int i) const
{
	return y_[i];
}

double Equation::get_yi(int i) const
{
	return yi_[i];
}

double Equation::get_yp(int i) const
{
	// a is a pointer to the first value in the derivative of f_ which is
	// stored in work_
	const double* a = &work_[99 + neqn_ * 4];
	return a[i];
}

void Equation::get_y(int i, int n, double* x) const
{
	const double* a = &y_[i];
	for(int j = 0; j < n; j++)
		*x++ = *a++;
}

void Equation::get_yi(int i, int n, double* x) const
{
	const double* a = &yi_[i];
	for (int j = 0; j < n; j++)
		*x++ = *a++;
}

void Equation::get_yp(int i, int n, double* x) const
{
	// a is a pointer to the first value in the derivative of f_ which is
	// stored in work_
	const double* a = &work_[99 + neqn_ * 4 + i];
	for (int j = 0; j < n; j++)
		*x++ = *a++;
}

void Equation::set_ti(double x)
{
	ti_ = x;
}

void Equation::set_yi(int i, double x)
{
	yi_[i] = x;
}

void Equation::set_yi(int i, int n, const double* x)
{
	double* a = &yi_[i];
	for (int j = 0; j < n; j++)
		*a++ = *x++;
}

void Equation::set_params(void* p)
{
	params_ = p;
}

// Returns error string associated with the current value of iflag_
std::string Equation::get_error_string(int iflag) const
{
	std::string error_string;

	switch(abs(iflag))
	{
		case 2:
			error_string = "ode returned iflag = 2, integration reached TOUT.\n";
			break;
		case 3:
			error_string = "ode returned iflag = 3, integration did not reach TOUT because the error tolerances were too small. But RELERR and ABSERR were increased appropriately for continuing.\n";
			break;
		case 4:
			error_string = "ode returned iflag = 4, integration did not reach TOUT because more than " + std::to_string(500 * max_iter) + " steps were taken.\n";
			break;
		case 5:
			error_string = "ode returned iflag = 5, integration did not reach TOUT because the equations appear to be stiff.\n";
			break;
		default:
			error_string = "ode returned iflag = " + std::to_string(iflag_) + ".\n";
	}

	return error_string;
}