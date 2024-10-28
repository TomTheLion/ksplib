#include <iostream>
#include "Equation.h"
#include "rk/rk.h"
#include "ode/wrapode.h"


Equation::Equation()
{
	method_ = Method::NONE;
	disp_ = false;
	max_iter_ = 0;
	tot_iter_ = 0;
	rej_iter_ = 0;
	iflag_ = 0;
	neqn_ = 0;
	reltol_ = 0.0;
	abstol_ = 0.0;
	t_ = 0.0;
	params_ = nullptr;
	f_ = nullptr;
}

Equation::Equation(
	void f(double t, double y[], double yp[], void* params),
	double t,
	const std::vector<double>& y,
	std::string method,
	double reltol,
	double abstol,
	void* params)
	: f_(f), neqn_(y.size()), t_(t), y_(y), reltol_(reltol), abstol_(abstol), params_(params)
{
	init_(method);
}

Equation::Equation(
	void f(double t, double y[], double yp[], void* params),
	int neqn,
	double t,
	const double* y,
	std::string method,
	double reltol,
	double abstol,
	void* params)
	: f_(f), neqn_(neqn), t_(t), y_(y, y + neqn), reltol_(reltol), abstol_(abstol), params_(params)
{
	init_(method);
}

void Equation::init_(std::string method)
{
	method_ = Method::NONE;
	disp_ = false;
	max_iter_ = 0;
	tot_iter_ = 0;
	rej_iter_ = 0;
	iflag_ = 0;

	if (method == "RK32")
	{
		method_ = Method::RK32;
		max_iter_ = 100000;
		rk::init(rk::method_rk32, f_, iflag_, neqn_, t_, y_, yp_, iwork_, work_, params_);
	}
	else if (method == "RK54")
	{
		method_ = Method::RK54;
		max_iter_ = 100000;
		rk::init(rk::method_rk54, f_, iflag_, neqn_, t_, y_, yp_, iwork_, work_, params_);
	}
	else if (method == "RK853")
	{
		method_ = Method::RK853;
		max_iter_ = 100000;
		rk::init(rk::method_rk853, f_, iflag_, neqn_, t_, y_, yp_, iwork_, work_, params_);
	}
	//else if (method == "VOMS")
	//{
	//	method_ = Method::VOMS;
	//	max_iter_ = 100000;
	//	wrap_ode::init(f_, iflag_, neqn_, t_, y_, yp_, iwork_, work_, params_);
	//}
	else
	{
		throw std::runtime_error("Equation failed to initialize, unknown method.");
	}
}

Equation::Equation(const Equation& equation)
{
	method_ = equation.method_;
	disp_ = equation.disp_;
	max_iter_ = equation.max_iter_;
	tot_iter_ = equation.tot_iter_;
	rej_iter_ = equation.rej_iter_;
	iflag_ = equation.iflag_;
	neqn_ = equation.neqn_;
	reltol_ = equation.reltol_;
	abstol_ = equation.abstol_;
	t_ = equation.t_;
	y_ = equation.y_;
	yp_ = equation.yp_;
	iwork_ = equation.iwork_;
	work_ = equation.work_;
	params_ = equation.params_;
	f_ = equation.f_;
}

Equation& Equation::operator = (const Equation& equation)
{
	if (&equation == this)
	{
		return *this;
	}

	method_ = equation.method_;
	disp_ = equation.disp_;
	max_iter_ = equation.max_iter_;
	tot_iter_ = equation.tot_iter_;
	rej_iter_ = equation.rej_iter_;
	iflag_ = equation.iflag_;
	neqn_ = equation.neqn_;
	reltol_ = equation.reltol_;
	abstol_ = equation.abstol_;
	t_ = equation.t_;
	y_ = equation.y_;
	yp_ = equation.yp_;
	iwork_ = equation.iwork_;
	work_ = equation.work_;
	params_ = equation.params_;
	f_ = equation.f_;

	return *this;
}

void Equation::step(double tout)
{
	step_(tout, 0.0, false);
}

void Equation::stepn(double tout)
{
	step_(tout, tout, true);
}

void Equation::stepn(double tout, double tlim)
{
	step_(tout, tlim, true);
}

void Equation::step_(double tout, double tlim, bool lim)
{
	iflag_ = lim ? -abs(iflag_) : abs(iflag_);
	switch (method_)
	{
	case Method::RK32:
		rk::step(rk::method_rk32, f_, disp_, max_iter_, tot_iter_, rej_iter_, iflag_, neqn_, reltol_, abstol_, t_, tout, tlim, y_, yp_, iwork_, work_, params_);
		break;
	case Method::RK54:
		rk::step(rk::method_rk54, f_, disp_, max_iter_, tot_iter_, rej_iter_, iflag_, neqn_, reltol_, abstol_, t_, tout, tlim, y_, yp_, iwork_, work_, params_);
		break;
	case Method::RK853:
		rk::step(rk::method_rk853, f_, disp_, max_iter_, tot_iter_, rej_iter_, iflag_, neqn_, reltol_, abstol_, t_, tout, tlim, y_, yp_, iwork_, work_, params_);
		break;
	//case Method::VOMS:
	//	wrap_ode::step(f_, max_iter_, tot_iter_, rej_iter_, iflag_, neqn_, reltol_, abstol_, t_, tout, y_, yp_, iwork_, work_, params_);
	//	break;
	default:
		break;
	}

	if (abs(iflag_) > 2)
	{
		std::cerr << get_error_string() << '\n';
	}
}

int Equation::get_tot_iter() const
{
	return tot_iter_;
}

int Equation::get_rej_iter() const
{
	return rej_iter_;
}

int Equation::get_iflag() const
{
	return iflag_;
}

int Equation::get_neqn() const
{
	return neqn_;
}

double Equation::get_t() const
{
	return t_;
}

double Equation::get_y(int i) const
{
	return y_[i];
}

std::vector<double> Equation::get_y() const
{
	return y_;
}

void Equation::get_y(int i, int n, double* x) const
{
	const double* a = &y_[i];
	for (int j = 0; j < n; j++)
		*x++ = *a++;
}

double Equation::get_yp(int i) const
{
	return yp_[i];
}

std::vector<double> Equation::get_yp() const
{
	return yp_;
}

void Equation::get_yp(int i, int n, double* x) const
{
	const double* a = &yp_[i];
	for (int j = 0; j < n; j++)
		*x++ = *a++;
}

std::string Equation::get_error_string() const
{
	std::string error_string;

	//switch (method_)
	//{
	//case Method::VOMS:
	//	switch (abs(iflag_))
	//	{
	//	case 2:
	//		error_string = "ode returned iflag = 2, integration reached TOUT.";
	//		break;
	//	case 3:
	//		error_string = "ode returned iflag = 3, integration did not reach TOUT because the error tolerances were too small. But RELERR and ABSERR were increased appropriately for continuing.";
	//		break;
	//	case 4:
	//		error_string = "ode returned iflag = 4, integration did not reach TOUT because more than " + std::to_string(max_iter_) + " steps were taken.";
	//		break;
	//	case 5:
	//		error_string = "ode returned iflag = 5, integration did not reach TOUT because the equations appear to be stiff.";
	//		break;
	//	default:
	//		error_string = "ode returned iflag = " + std::to_string(iflag_) + ".";
	//	}
	//	break;
	//default:
		switch (abs(iflag_))
		{
		case 2:
			error_string = "rk returned iflag = 2, integration reached TOUT.";
			break;
		case 3:
			error_string = "rk returned iflag = 3, integration did not reach TOUT because the error tolerances were too small.";
			break;
		case 4:
			error_string = "rk returned iflag = 4, integration did not reach TOUT because more than " + std::to_string(max_iter_) + " steps were taken.";
			break;
		case 5:
			error_string = "rk returned iflag = 5, integration did not reach TOUT because TLIM was incompatible with integration direction.";
			break;
		default:
			error_string = "rk returned iflag = " + std::to_string(iflag_) + ".";
		}
	//}

	return error_string;
}

void Equation::set_disp(bool disp)
{
	disp_ = disp;
}