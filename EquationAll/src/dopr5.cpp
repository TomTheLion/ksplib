#include "dopr5.h"

namespace dopr5
{
	// Machine precision
	constexpr double eps = 2.220446049250313E-016;

	// Node coefficients
	constexpr double c1 = 1.0 / 5.0;
	constexpr double c2 = 3.0 / 10.0;
	constexpr double c3 = 4.0 / 5.0;
	constexpr double c4 = 8.0 / 9.0;

	// Runge-Kutta matrix
	constexpr double a11 = 1.0 / 5.0;
	constexpr double a21 = 3.0 / 40.0;
	constexpr double a22 = 9.0 / 40.0;
	constexpr double a31 = 44.0 / 45.0;
	constexpr double a32 = -56.0 / 15.0;
	constexpr double a33 = 32.0 / 9.0;
	constexpr double a41 = 19372.0 / 6561.0;
	constexpr double a42 = -25360.0 / 2187.0;
	constexpr double a43 = 64448.0 / 6561.0;
	constexpr double a44 = -212.0 / 729.0;
	constexpr double a51 = 9017.0 / 3168.0;
	constexpr double a52 = -355.0 / 33.0;
	constexpr double a53 = 46732.0 / 5247.0;
	constexpr double a54 = 49.0 / 176.0;
	constexpr double a55 = -5103.0 / 18656.0;
	constexpr double a61 = 35.0 / 384.0;
	constexpr double a63 = 500.0 / 1113.0;
	constexpr double a64 = 125.0 / 192.0;
	constexpr double a65 = -2187.0 / 6784.0;
	constexpr double a66 = 11.0 / 84.0;

	// Error coefficients
	constexpr double e1 = 71.0 / 57600.0;
	constexpr double e3 = -71.0 / 16695.0;
	constexpr double e4 = 71.0 / 1920.0;
	constexpr double e5 = -17253.0 / 339200.0;
	constexpr double e6 = 22.0 / 525.0;
	constexpr double e7 = -1.0 / 40.0;

	// Interpolation coefficients
	constexpr double d1 = -12715105075.0 / 11282082432.0;
	constexpr double d3 = 87487479700.0 / 32700410799.0;
	constexpr double d4 = -10690763975.0 / 1880347072.0;
	constexpr double d5 = 701980252875.0 / 199316789632.0;
	constexpr double d6 = -1453857185.0 / 822651844.0;
	constexpr double d7 = 69997945.0 / 29380423.0;

	// Tests if a value falls within a range
	// a = value to be tested
	// r1, r2 = bounds of the range
	bool in_range(double a, double r1, double r2)
	{
		return r1 <= r2 ? r1 <= a && a <= r2 : r2 <= a && a <= r1;
	}

	// Initializes Equation
	// f = pointer to function that calculates the derivative of the problem
	// iflag = flag which holds the return status of the integrator
	// neqn = number of equations
	// t = initial time
	// y = initial state of the problem
	// yp = derivative of problem
	// iwork = internal work space for the integrator
	// work = internal work space for the integrator
	// params = optional pointer to parameters for f
	void init(
		void f(double t, double y[], double yp[], void* params),
		int& iflag,
		int neqn,
		double t,
		std::vector<double>& y,
		std::vector<double>& yp,
		std::vector<int>& iwork,
		std::vector<double>& work,
		void* params)
	{
		// iflag of 1 indicates integrator is in its initial state
		iflag = 1;
		// set the size of yp and the internal workspace
		yp.resize(neqn);
		iwork.resize(1);
		work.resize(5 + 14 * neqn);
		// set the initial value of errold, t, and internal copies of the problem state
		// and its derivative
		double& errold = work[0];
		double& tt = work[3];
		double* yy = &work[5];
		double* yyp = yy + neqn;
		errold = 1.0;
		tt = t;
		f(t, y.data(), yp.data(), params);
		for (int i = 0; i < neqn; i++)
		{
			yy[i] = y[i];
			yyp[i] = yp[i];
		}
	}

	// Steps Equation from t to tout
	// f = pointer to function that calculates the derivative of the problem
	// max_iter = maximum number of iterations
	// iflag = flag which holds the return status of the integrator
	// neqn = number of equations
	// reltol = relative error tolerance
	// abstol = absolute error tolerance
	// t = initial time
	// tout = final desired integration time
	// y = initial state of the problem
	// yp = derivative of problem
	// iwork = internal work space for the integrator
	// work = internal work space for the integrator
	// params = optional pointer to parameters for f
	void step(
		void f(double t, double y[], double yp[], void* params),
		int max_iter,
		int& iflag,
		int neqn,
		double reltol,
		double abstol,
		double& t,
		double tout,
		std::vector<double>& y,
		std::vector<double>& yp,
		std::vector<int>& iwork,
		std::vector<double>& work,
		void* params)
	{
		int& reject = iwork[0];

		double& errold = work[0];
		double& d = work[1];
		double& hh = work[2];
		double& tt = work[3];
		double& tw = work[4];
		double tti = tt;

		double* yy = &work[5];
		double* yyp = yy + neqn;
		double* yw = yyp + neqn;
		double* k2 = yw + neqn;
		double* k3 = k2 + neqn;
		double* k4 = k3 + neqn;
		double* k5 = k4 + neqn;
		double* k6 = k5 + neqn;
		double* k7 = k6 + neqn;
		double* r1 = k7 + neqn;
		double* r2 = r1 + neqn;
		double* r3 = r2 + neqn;
		double* r4 = r3 + neqn;
		double* r5 = r4 + neqn;

		// if integrator is in initial state, set initial stepsize
		// if integrator was previously successful and tout falls within the last
		// step interpolate and return
		if (iflag == 1)
		{
			initial_step_size(neqn, reltol, abstol, hh, yy, yyp);
		}
		else if (iflag == 2 && in_range(tout, tt, tw))
		{
			intrp(neqn, t, tout, y, yp, tt, tw, r1, r2, r3, r4, r5);
			return;
		}

		// set direction of integration
		d = tout > tt ? 1.0 : -1.0;

		// main integration loop
		for (int iter = 0; iter < max_iter; iter++)
		{
			if (hh < 4.0 * eps * abs(tt - tti))
			{
				// iflag of 3 indicates that the error tolerances are too low
				iflag = 3;
			}

			// perform step then estimate error and update step size
			dy(f, neqn, d * hh, tt, yy, yyp, yw, k2, k3, k4, k5, k6, k7, params);
			update_step_size(neqn, reltol, abstol, reject, errold, d, hh, tt, tw, yy, yyp, yw, k3, k4, k5, k6, k7);

			// test if step was successful
			if (reject)
				continue;

			// if step was successful and tout falls within the step prepare dense
			// output, interpolate, and return, if step was successful and tout falls
			// outside the step prepare the next step
			if (in_range(tout, tt, tw))
			{
				// iflag of 2 indicates that integration was successful
				iflag = 2;
				dense(neqn, d * hh, yy, yyp, yw, k3, k4, k5, k6, k7, r1, r2, r3, r4, r5);
				double temp = tt;
				tt = tw;
				tw = temp;
				for (int i = 0; i < neqn; i++)
				{
					yy[i] = yw[i];
					yyp[i] = k7[i];
				}
				intrp(neqn, t, tout, y, yp, tt, tw, r1, r2, r3, r4, r5);
				return;
			}
			else
			{
				tt = tw;
				for (int i = 0; i < neqn; i++)
				{
					yy[i] = yw[i];
					yyp[i] = k7[i];
				}
			}
		}

		// iflag of 4 indicates that the maximum number of iterations was exceeded
		iflag = 4;
	}

	// Calculates Runge-Kutta steps
	// f = pointer to function that calculates the derivative of the problem
	// neqn = number of equations
	// h = current step size with direction
	// tt = initial time
	// yy = current state of the problem
	// yyp = current derivative of problem
	// yw = working state of the problem
	// k = Runge-kutta slope estimates
	// params = optional pointer to parameters for f
	void dy(
		void f(double t, double y[], double yp[], void* params),
		int neqn,
		double h,
		double tt,
		double* yy,
		double* yyp,
		double* yw,
		double* k2,
		double* k3,
		double* k4,
		double* k5,
		double* k6,
		double* k7,
		void* params)
	{
		for (int i = 0; i < neqn; i++)
			yw[i] = yy[i] + h * (a11 * yyp[i]);
		f(tt + c1 * h, yw, k2, params);

		for (int i = 0; i < neqn; i++)
			yw[i] = yy[i] + h * (a21 * yyp[i] + a22 * k2[i]);
		f(tt + c2 * h, yw, k3, params);

		for (int i = 0; i < neqn; i++)
			yw[i] = yy[i] + h * (a31 * yyp[i] + a32 * k2[i] + a33 * k3[i]);
		f(tt + c3 * h, yw, k4, params);

		for (int i = 0; i < neqn; i++)
			yw[i] = yy[i] + h * (a41 * yyp[i] + a42 * k2[i] + a43 * k3[i] + a44 * k4[i]);
		f(tt + c4 * h, yw, k5, params);

		for (int i = 0; i < neqn; i++)
			yw[i] = yy[i] + h * (a51 * yyp[i] + a52 * k2[i] + a53 * k3[i] + a54 * k4[i] + a55 * k5[i]);
		f(tt + h, yw, k6, params);

		for (int i = 0; i < neqn; i++)
			yw[i] = yy[i] + h * (a61 * yyp[i] + a63 * k3[i] + a64 * k4[i] + a65 * k5[i] + a66 * k6[i]);
		f(tt + h, yw, k7, params);
	}

	// Estimates the initial step size
	// neqn = number of equations
	// reltol = relative error tolerance
	// abstol = absolute error tolerance
	// hh = current step size
	// yy = current state of the problem
	// yyp = current derivative of problem
	void initial_step_size(
		int neqn, 
		double reltol, 
		double abstol, 
		double& hh, 
		double* yy, 
		double* yyp)
	{
		double err = 0.0;

		for (int i = 0; i < neqn; i++)
		{
			double sci = abstol + reltol * abs(yy[i]);
			double ei = yyp[i];
			err += pow(ei / sci, 2.0);
		}

		hh = pow(err / neqn, -0.10);
	}

	// Estimates error and calculates next step size
	// neqn = number of equations
	// reltol = relative error tolerance
	// abstol = absolute error tolerance
	// reject = indicates if step was rejected
	// errold = err from previous step
	// d = direction of integration
	// hh = current step size
	// tt = initial time
	// tw = working time
	// yy = current state of the problem
	// yyp = current derivative of problem
	// yw = working state of the problem
	// k = Runge-kutta slope estimates
	void update_step_size(
		int neqn,
		double reltol,
		double abstol,
		int& reject,
		double& errold,
		double d,
		double& hh,
		double tt,
		double& tw,
		double* yy,
		double* yyp,
		double* yw,
		double* k3,
		double* k4,
		double* k5,
		double* k6,
		double* k7)
	{
		// step size control parameters
		const double alpha = 0.17;
		const double beta = 0.03;
		const double safe = 0.92;
		const double min_scale = 0.2;
		const double max_scale = 10.0;

		// esimate error
		double scale;
		double err = 0.0;

		for (int i = 0; i < neqn; i++)
		{
			double sci = abstol + reltol * std::max(abs(yy[i]), abs(yw[i]));
			double ei = hh * (e1 * yyp[i] + e3 * k3[i] + e4 * k4[i] + e5 * k5[i] + e6 * k6[i] + e7 * k7[i]);
			err += pow(ei / sci, 2.0);
		}

		err = sqrt(err / neqn);

		// if err < 1.0 accept step, update tw, hh, errold, and set reject to 0
		// if err > 1.0 reject step, update hh, and set reject to 1
		if (err < 1.0)
		{
			if (err == 0.0)
			{
				scale = max_scale;
			}
			else
			{
				scale = safe * pow(err, -alpha) * pow(errold, beta);
				if (scale < min_scale) { scale = min_scale; }
				if (scale > max_scale) { scale = max_scale; }
			}
			if (reject)
			{
				tw = tt + d * hh;
				hh *= std::min(scale, 1.0);
			}
			else
			{
				tw = tt + d * hh;
				hh *= scale;
			}
			errold = std::max(err, 1e-4);
			reject = 0;
		}
		else
		{
			scale = std::max(safe * pow(err, -alpha), min_scale);
			hh *= scale;
			reject = 1;
		}
	}

	// Calculates coefficients for dense output
	// neqn = number of equations
	// h = current step size with direction
	// yy = current state of the problem
	// yyp = current derivative of problem
	// yw = working state of the problem
	// k = Runge-kutta slope estimates
	// r = dense output coefficients
	void dense(
		int neqn,
		double h,
		double* yy,
		double* yyp,
		double* yw,
		double* k3,
		double* k4,
		double* k5,
		double* k6,
		double* k7,
		double* r1,
		double* r2,
		double* r3,
		double* r4,
		double* r5)
	{
		for (int i = 0; i < neqn; i++)
		{
			r1[i] = yy[i];
			double dy = yw[i] - yy[i];
			double bspl = h * yyp[i] - dy;
			r2[i] = dy;
			r3[i] = bspl;
			r4[i] = dy - h * k7[i] - bspl;
			r5[i] = h * (d1 * yyp[i] + d3 * k3[i] + d4 * k4[i] + d5 * k5[i] + d6 * k6[i] + d7 * k7[i]);



		}
	}

	// Interpolates dense output
	// neqn = number of equations
	// t = initial time
	// tout = final desired integration time
	// y = initial state of the problem
	// yp = derivative of problem
	// tt = initial time
	// tw = working time
	// r = dense output coefficients
	void intrp(
		int neqn,
		double& t,
		double tout,
		std::vector<double>& y,
		std::vector<double>& yp,
		double tt,
		double tw,
		double* r1,
		double* r2,
		double* r3,
		double* r4,
		double* r5)
	{
		t = tout;
		double h = (tt - tw);
		double s = (tout - tw) / h;
		double s1 = 1.0 - s;
		double s2 = s1 - s;
		double s3 = s * (s1 + s2);
		double s4 = 2.0 * s * s1 * s2;

		for (int i = 0; i < neqn; i++)
		{
			y[i] = r1[i] + s * (r2[i] + s1 * (r3[i] + s * (r4[i] + s1 * r5[i])));
			yp[i] = 1.0 / h * (r2[i] + s2 * r3[i] + s3 * r4[i] + s4 * r5[i]);
		}
	}
}