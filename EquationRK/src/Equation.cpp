#include <iostream>
#include <iomanip>
#include <string>

#include "Equation.h"

constexpr double eps_ = 2.220446049250313E-016;

constexpr double a1 = 1.0 / 5.0;
constexpr double a2 = 3.0 / 10.0;
constexpr double a3 = 4.0 / 5.0;
constexpr double a4 = 8.0 / 9.0;

constexpr double b11 = 1.0 / 5.0;
constexpr double b21 = 3.0 / 40.0;
constexpr double b22 = 9.0 / 40.0;
constexpr double b31 = 44.0 / 45.0;
constexpr double b32 = -56.0 / 15.0;
constexpr double b33 = 32.0 / 9.0;
constexpr double b41 = 19372.0 / 6561.0;
constexpr double b42 = -25360.0 / 2187.0;
constexpr double b43 = 64448.0 / 6561.0;
constexpr double b44 = -212.0 / 729.0;
constexpr double b51 = 9017.0 / 3168.0;
constexpr double b52 = -355.0 / 33.0;
constexpr double b53 = 46732.0 / 5247.0;
constexpr double b54 = 49.0 / 176.0;
constexpr double b55 = -5103.0 / 18656.0;
constexpr double b61 = 35.0 / 384.0;
constexpr double b63 = 500.0 / 1113.0;
constexpr double b64 = 125.0 / 192.0;
constexpr double b65 = -2187.0 / 6784.0;
constexpr double b66 = 11.0 / 84.0;

constexpr double e1 = 71.0 / 57600.0;
constexpr double e3 = -71.0 / 16695.0;
constexpr double e4 = 71.0 / 1920.0;
constexpr double e5 = -17253.0 / 339200.0;
constexpr double e6 = 22.0 / 525.0;
constexpr double e7 = -1.0 / 40.0;

constexpr double d1 = -12715105075.0 / 11282082432.0;
constexpr double d3 = 87487479700.0 / 32700410799.0;
constexpr double d4 = -10690763975.0 / 1880347072.0;
constexpr double d5 = 701980252875.0 / 199316789632.0;
constexpr double d6 = -1453857185.0 / 822651844.0;
constexpr double d7 = 69997945.0 / 29380423.0;

// Create an uninitialized Equation object
Equation::Equation()
{
	max_iter_ = 0;
	reject_ = false;
	neqn_ = 0;
	abstol_ = 0.0;
	reltol_ = 0.0;
	errold_ = 0.0;
	d_ = 0.0;
	h_ = 0.0;
	t_ = 0.0;
	ti_ = 0.0;
	tintrp_ = 0.0;

	params_ = nullptr;
	f_ = nullptr;
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
	double reltol,
	double abstol,
	void* params)
	: f_(f), neqn_(neqn), t_(t), ti_(t), tintrp_(t), reltol_(reltol), abstol_(abstol), params_(params)
{
	max_iter_ = 10000000000;
	reject_ = false;
	errold_ = 1.0;
	d_ = 1.0;
	h_ = 0.0;

	y_.resize(neqn_);
	yi_.resize(neqn_);
	yp_.resize(neqn_);

	yw_.resize(neqn_);
	yintrp_.resize(neqn_);
	ypintrp_.resize(neqn_);

	k2_.resize(neqn_);
	k3_.resize(neqn_);
	k4_.resize(neqn_);
	k5_.resize(neqn_);
	k6_.resize(neqn_);
	k7_.resize(neqn_);
	r1_.resize(neqn_);
	r2_.resize(neqn_);
	r3_.resize(neqn_);
	r4_.resize(neqn_);
	r5_.resize(neqn_);

	for (int i = 0; i < neqn_; i++)
	{
		y_[i] = y[i];
		yi_[i] = y[i];
	}

	f_(t_, y_.data(), yp_.data(), params_);
}

// Create a copy of another Equation object
Equation::Equation(const Equation& equation)
{
	max_iter_ = equation.max_iter_;
	reject_ = equation.reject_;
	neqn_ = equation.neqn_;
	reltol_ = equation.reltol_;
	abstol_ = equation.abstol_;
	errold_ = equation.errold_;
	d_ = equation.d_;
	h_ = equation.h_;
	t_ = equation.t_;
	ti_ = equation.ti_;
	tintrp_ = equation.tintrp_;

	y_.resize(neqn_);
	yi_.resize(neqn_);
	yp_.resize(neqn_);
	yw_.resize(neqn_);
	yintrp_.resize(neqn_);
	ypintrp_.resize(neqn_);

	k2_.resize(neqn_);
	k3_.resize(neqn_);
	k4_.resize(neqn_);
	k5_.resize(neqn_);
	k6_.resize(neqn_);
	k7_.resize(neqn_);
	r1_.resize(neqn_);
	r2_.resize(neqn_);
	r3_.resize(neqn_);
	r4_.resize(neqn_);
	r5_.resize(neqn_);

	params_ = equation.params_;
	f_ = equation.f_;

	for (int i = 0; i < neqn_; i++)
	{
		y_[i] = equation.y_[i];
		yi_[i] = equation.yi_[i];
		yp_[i] = equation.yp_[i];
		yintrp_[i] = equation.yintrp_[i];
		ypintrp_[i] = equation.ypintrp_[i];
		r1_[i] = equation.r1_[i];
		r2_[i] = equation.r2_[i];
		r3_[i] = equation.r3_[i];
		r4_[i] = equation.r4_[i];
		r5_[i] = equation.r5_[i];
	}
}

// Assign state from an existing Equation object
Equation& Equation::operator = (const Equation& equation)
{
	if (&equation == this)
	{
		return *this;
	}

	max_iter_ = equation.max_iter_;
	neqn_ = equation.neqn_;
	reltol_ = equation.reltol_;
	abstol_ = equation.abstol_;
	errold_ = equation.errold_;
	d_ = equation.d_;
	h_ = equation.h_;
	t_ = equation.t_;
	ti_ = equation.ti_;
	tintrp_ = equation.tintrp_;

	y_.resize(neqn_);
	yi_.resize(neqn_);
	yp_.resize(neqn_);
	yw_.resize(neqn_);
	yintrp_.resize(neqn_);
	ypintrp_.resize(neqn_);

	k2_.resize(neqn_);
	k3_.resize(neqn_);
	k4_.resize(neqn_);
	k5_.resize(neqn_);
	k6_.resize(neqn_);
	k7_.resize(neqn_);
	r1_.resize(neqn_);
	r2_.resize(neqn_);
	r3_.resize(neqn_);
	r4_.resize(neqn_);
	r5_.resize(neqn_);

	params_ = equation.params_;
	f_ = equation.f_;


	for (int i = 0; i < neqn_; i++)
	{
		y_[i] = equation.y_[i];
		yi_[i] = equation.yi_[i];
		yp_[i] = equation.yp_[i];
		yintrp_[i] = equation.yintrp_[i];
		ypintrp_[i] = equation.ypintrp_[i];
		r1_[i] = equation.r1_[i];
		r2_[i] = equation.r2_[i];
		r3_[i] = equation.r3_[i];
		r4_[i] = equation.r4_[i];
		r5_[i] = equation.r5_[i];
	}

	return *this;
}

// Destroy equation object
Equation::~Equation()
{

}

void Equation::step(double tout)
{
	d_ = 1.0 ? tout > t_ : -1.0;

	if (d_ * (tout - t_) < h_)
	{
		intrp_(tout);
		return;
	}

	if (h_ == 0)
	{
		set_initial_step_size_();
	}

	for (int iter = 0; iter < max_iter_; iter++)
	{
		if (h_ < 4.0 * eps_ * (t_ - ti_))
		{
			throw std::runtime_error("Equation failed, minimum step size reached.");
		}

		// std::cout << std::setprecision(17) << h_ << '\n';
		dopr5_();
		double htemp = h_;
		update_step_size_();

		if (reject_)
			continue;

		if (d_ * (tout - t_) < h_)
		{
			dense_();
			intrp_(tout);
			t_ += htemp;
			std::swap(y_, yw_);
			std::swap(yp_, k7_);
			return;
		}
		else
		{
			t_ += htemp;
			std::swap(y_, yw_);
			std::swap(yp_, k7_);
		}
	}

	throw std::runtime_error("Equation failed, maximum iterations reached.");
}

// Resets Equation to its initial values
void Equation::reset()
{
	errold_ = 0.0;
	h_ = 0.0;
	t_ = ti_;

	for (int i = 0; i < neqn_; i++)
	{
		y_[i] = yi_[i];
	}

	f_(t_, y_.data(), yp_.data(), params_);
}

void Equation::set_max_iter(int n)
{
	max_iter_ = n;
}

int Equation::get_max_iter() const
{
	return max_iter_;
}

int Equation::get_neqn() const
{
	return neqn_;
}

double Equation::get_t() const
{
	return tintrp_;
}

double Equation::get_ti() const
{
	return ti_;
}

double Equation::get_y(int i) const
{
	return yintrp_[i];
}

double Equation::get_yi(int i) const
{
	return yi_[i];
}

double Equation::get_yp(int i) const
{
	return ypintrp_[i];
}

void Equation::get_y(int i, int n, double* x) const
{
	const double* a = &yintrp_[i];
	for (int j = 0; j < n; j++)
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
	const double* a = &ypintrp_[i];
	for (int j = 0; j < n; j++)
		*x++ = *a++;
}

void Equation::dopr5_()
{
	double h = d_ * h_;

	for (int i = 0; i < neqn_; i++)
		yw_[i] = y_[i] + h * (b11 * yp_[i]);
	f_(t_ + a1 * h, yw_.data(), k2_.data(), params_);

	for (int i = 0; i < neqn_; i++)
		yw_[i] = y_[i] + h * (b21 * yp_[i] + b22 * k2_[i]);
	f_(t_ + a2 * h, yw_.data(), k3_.data(), params_);

	for (int i = 0; i < neqn_; i++)
		yw_[i] = y_[i] + h * (b31 * yp_[i] + b32 * k2_[i] + b33 * k3_[i]);
	f_(t_ + a3 * h, yw_.data(), k4_.data(), params_);

	for (int i = 0; i < neqn_; i++)
		yw_[i] = y_[i] + h * (b41 * yp_[i] + b42 * k2_[i] + b43 * k3_[i] + b44 * k4_[i]);
	f_(t_ + a4 * h, yw_.data(), k5_.data(), params_);

	for (int i = 0; i < neqn_; i++)
		yw_[i] = y_[i] + h * (b51 * yp_[i] + b52 * k2_[i] + b53 * k3_[i] + b54 * k4_[i] + b55 * k5_[i]);
	f_(t_ + h, yw_.data(), k6_.data(), params_);

	for (int i = 0; i < neqn_; i++)
		yw_[i] = y_[i] + h * (b61 * yp_[i] + b63 * k3_[i] + b64 * k4_[i] + b65 * k5_[i] + b66 * k6_[i]);
	f_(t_ + h, yw_.data(), k7_.data(), params_);
}

void Equation::set_initial_step_size_()
{
	double err = 0.0;

	for (int i = 0; i < neqn_; i++)
	{
		double si = (1.0 + reltol_ / abstol_ * abs(y_[i]));
		err = std::max(err, abs(yp_[i]) / si);
	}

	h_ = pow(abstol_ / err, 1.0 / 6.0);
}

void Equation::update_step_size_()
{
	static const double alpha = 0.17;
	static const double beta = 0.03;
	static const double safe = 0.92;
	static const double min_scale = 0.2;
	static const double max_scale = 10.0;

	double scale;
	double err = 0.0;

	for (int i = 0; i < neqn_; i++)
	{
		double sk = abstol_ + reltol_ * std::max(abs(y_[i]), abs(yw_[i]));
		double ei = h_ * (e1 * yp_[i] + e3 * k3_[i] + e4 * k4_[i] + e5 * k5_[i] + e6 * k6_[i] + e7 * k7_[i]);
		err += pow(ei / sk, 2.0);
	}

	err = sqrt(err / neqn_);

	if (err < 1.0)
	{
		if (err == 0.0)
		{
			scale = max_scale;
		}
		else
		{
			scale = safe * pow(err, -alpha) * pow(errold_, beta);
			if (scale < min_scale) { scale = min_scale; }
			if (scale > max_scale) { scale = max_scale; }
		}
		if (reject_)
		{
			h_ *= std::min(scale, 1.0);
		}
		else
		{
			h_ *= scale;
		}
		errold_ = std::max(err, 1e-4);
		reject_ = false;
	}
	else
	{
		scale = std::max(safe * pow(err, -alpha), min_scale);
		h_ *= scale;
		reject_ = true;
	}
}

void Equation::dense_()
{
	double h = d_ * h_;

	for (int i = 0; i < neqn_; i++)
	{
		r1_[i] = y_[i];
		double dy = yw_[i] - y_[i];
		double bspl = h * yp_[i] - dy;
		r2_[i] = dy;
		r3_[i] = bspl;
		r4_[i] = dy - h * k7_[i] - bspl;
		r5_[i] = h * (
			  d1 * yp_[i]
			+ d3 * k3_[i]
			+ d4 * k4_[i]
			+ d5 * k5_[i]
			+ d6 * k6_[i]
			+ d7 * k7_[i]);
	}
}

void Equation::intrp_(double tout)
{
	tintrp_ = tout;
	double h = d_ * h_;
	double s = (tout - t_) / h;
	double s1 = 1.0 - s;
	double s2 = s1 - s;
	double s3 = s * (s1 + s2);
	double s4 = 2.0 * s * s1 * s2;

	for (int i = 0; i < neqn_; i++)
	{
		double a = r5_[i];
		double b = -(r4_[i] + 2.0 * r5_[i]);
		double c = -(r3_[i] - r4_[i] - r5_[i]);
		double d = r2_[i] + r3_[i];
		double e = r1_[i];
		yintrp_[i] = r1_[i] + s * (r2_[i] + s1 * (r3_[i] + s * (r4_[i ] + s1 * r5_[i])));
		ypintrp_[i] = 1.0 / h * (r2_[i] + s2 * r3_[i] + s3 * r4_[i] + s4 * r5_[i]);
	}
}