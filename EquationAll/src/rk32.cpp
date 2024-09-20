#include "rk32.h"

namespace rk32
{
	// Machine precision
	constexpr double eps = 2.220446049250313E-016;

	// Node coefficients
	constexpr double c2 = 1.0 / 2.0;
	constexpr double c3 = 3.0 / 4.0;

	// Runge-Kutta matrix
	constexpr double a21 = 1.0 / 2.0;
	constexpr double a32 = 3.0 / 4.0;
	constexpr double a41 = 2.0 / 9.0;
	constexpr double a42 = 1.0 / 3.0;
	constexpr double a43 = 4.0 / 9.0;

	// Error coefficients
	constexpr double e1 = 5.0 / 72.0;
	constexpr double e2 = -1.0 / 12.0;
	constexpr double e3 = -1.0 / 9.0;
	constexpr double e4 = 1.0 / 8.0;

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
		work.resize(5 + 10 * neqn);
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
		int& tot_iter,
		int& rej_iter,
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
		double* r1 = k4 + neqn;
		double* r2 = r1 + neqn;
		double* r3 = r2 + neqn;
		double* r4 = r3 + neqn;

		// if integrator is in initial state, set initial stepsize
		// if integrator was previously successful and tout falls within the last
		// step interpolate and return
		if (iflag == 1)
		{
			initial_step_size(neqn, reltol, abstol, hh, yy, yyp);
		}
		else if (iflag == 2 && in_range(tout, tt, tw))
		{
			intrp(neqn, t, tout, y, yp, tt, tw, r1, r2, r3, r4);
			return;
		}

		// set direction of integration
		d = tout >= tt ? 1.0 : -1.0;

		// main integration loop
		for (int iter = 0; iter < max_iter; iter++)
		{
			tot_iter++;

			if (hh < 4.0 * eps * abs(tt - tti))
			{
				// iflag of 3 indicates that the error tolerances are too low
				iflag = 3;
			}

			// perform step then estimate error and update step size
			dy(f, neqn, d * hh, tt, yy, yyp, yw, k2, k3, k4, params);
			update_step_size(neqn, reltol, abstol, reject, errold, d, hh, tt, tw, yy, yyp, yw, k2, k3, k4);

			// test if step was successful
			if (reject)
			{
				rej_iter++;
				continue;
			}

			// if step was successful and tout falls within the step prepare dense
			// output, interpolate, and return, if step was successful and tout falls
			// outside the step prepare the next step
			if (in_range(tout, tt, tw))
			{
				// iflag of 2 indicates that integration was successful
				iflag = 2;
				dense(neqn, tt, tw, yy, yyp, yw, k2, k3, k4, r1, r2, r3, r4);
				double temp = tt;
				tt = tw;
				tw = temp;
				for (int i = 0; i < neqn; i++)
				{
					yy[i] = yw[i];
					yyp[i] = k4[i];
				}
				intrp(neqn, t, tout, y, yp, tt, tw, r1, r2, r3, r4);
				return;
			}
			else
			{
				tt = tw;
				for (int i = 0; i < neqn; i++)
				{
					yy[i] = yw[i];
					yyp[i] = k4[i];
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
		void* params)
	{
		for (int i = 0; i < neqn; i++)
			yw[i] = yy[i] + h * (a21 * yyp[i]);
		f(tt + c2 * h, yw, k2, params);

		for (int i = 0; i < neqn; i++)
			yw[i] = yy[i] + h * (a32 * k2[i]);
		f(tt + c3 * h, yw, k3, params);

		for (int i = 0; i < neqn; i++)
			yw[i] = yy[i] + h * (a41 * yyp[i] + a42 * k2[i] + a43 * k3[i]);
		f(tt + h, yw, k4, params);
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

		hh = pow(err / neqn, -(1.0 / 6.0));
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
		double* k2,
		double* k3,
		double* k4)
	{
		// step size control parameters
		const double alpha = 0.33;
		const double beta = 0.0;
		const double safe = 0.90;
		const double min_scale = 0.2;
		const double max_scale = 10.0;

		// esimate error
		double scale;
		double err = 0.0;

		for (int i = 0; i < neqn; i++)
		{
			double sci = abstol + reltol * std::max(abs(yy[i]), abs(yw[i]));
			double ei = hh * (e1 * yyp[i] + e2 * k2[i] + e3 * k3[i] + e4 * k4[i]);
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
	// tt = initial time
	// tw = working time
	// yy = current state of the problem
	// yyp = current derivative of problem
	// yw = working state of the problem
	// k = Runge-kutta slope estimates
	// r = dense output coefficients
	void dense(
		int neqn,
		double tt,
		double tw,
		double* yy,
		double* yyp,
		double* yw,
		double* k2,
		double* k3,
		double* k4,
		double* r1,
		double* r2,
		double* r3,
		double* r4)
	{
		double h = tw - tt;
		for (int i = 0; i < neqn; i++)
		{
			r1[i] = yy[i];
			r2[i] = yw[i] - yy[i];
			r3[i] = h * yyp[i] - r2[i];
			r4[i] = r2[i] - h * k4[i] - r3[i];
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
		double* r4)
	{
		t = tout;
		double h = tt - tw;
		double s = (tout - tw) / h;
		double s1 = 1.0 - s;

		for (int i = 0; i < neqn; i++)
		{
			double a2 = r3[i] + r4[i] * s;
			double a1 = r2[i] + a2 * s1;

			y[i] = r1[i] + s * a1;
			yp[i] = 1.0 / h * (a1 - s * (a2 - s1 * r4[i]));
		}
	}
}