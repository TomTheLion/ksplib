#include <iostream>
#include <iomanip>
#include "rk.h"
#include "rk32.h"
#include "rk54.h"
#include "rk853.h"

namespace rk
{
	// Machine precision
	constexpr double eps = 2.220446049250313E-016;

	// Method definitions
	struct RKMethod method_rk32 {
		true, 3, 10, 5, 0.33, 0.0, 0.9,
		rk32::dy,
		rk32::error,
		rk32::dense,
		rk32::intrp};

	struct RKMethod method_rk54 {
		true, 5, 14, 8, 0.20, 0.0, 0.9,
		rk54::dy,
		rk54::error,
		rk54::dense,
		rk54::intrp
	};

	struct RKMethod method_rk853 {
		false, 8, 22, 4, 0.125, 0.0, 0.9,
		rk853::dy,
		rk853::error,
		rk853::dense,
		rk853::intrp
	};

	// Tests if a value falls within a range
	bool in_range(double a, double r1, double r2)
	{
		return r1 <= r2 ? r1 <= a && a <= r2 : r2 <= a && a <= r1;
	}

	// Initializes Equation
	// sets initial value of iflag, initializes workspace memory, copies initial
	// problem state, and calculates initial derivatives
	void init(
		RKMethod method,
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
		iflag = 1;
		yp.resize(neqn);
		iwork.resize(1);
		work.resize(5 + method.swork * neqn);
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
	void step(
		RKMethod method,
		void f(double t, double y[], double yp[], void* params),
		bool disp,
		int max_iter,
		int& tot_iter,
		int& rej_iter,
		int& iflag,
		int neqn,
		double reltol,
		double abstol,
		double& t,
		double tout,
		double tlim,
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

		double* yy = &work[5];
		double* yyp = yy + neqn;

		double* yw = yyp + neqn;
		double* ywp = &work[5 + method.ywp_offset * neqn];

		bool last = false;
		double tti = tt;

		// if integrator is in initial state, set initial stepsize
		// if integrator was previously successful and tout falls within the last
		// step interpolate and return
		if (abs(iflag) == 1)
		{
			initial_step_size(method.order, neqn, reltol, abstol, hh, yy, yyp);
		}
		else if (abs(iflag) == 2 && in_range(tout, tt, tw))
		{
			method.intrp(neqn, t, tout, y, yp, work);
			return;
		}

		if (t == tout)
		{
			return;
		}

		// set direction of integration
		d = tout > tt ? 1.0 : -1.0;

		// main integration loop
		for (int iter = 0; iter < max_iter; iter++)
		{
			tot_iter++;

			if (hh < 4.0 * eps * abs(tt - tti))
			{
				// iflag of 3 indicates that the error tolerances are too low
				iflag = 3;
				return;
			}

			// if iflag < 0 and if tt + hh would exceed tlim then limit hh
			if (iflag < 0 && hh > d * (tlim - tt))
			{
				last = true;
				hh = d * (tlim - tt);
			}
			else
			{
				last = false;
			}

			// perform step then estimate error, display, and update step size
			method.dy(f, neqn, work, params);
			tw = tt + d * hh;
			double err = method.error(neqn, reltol, abstol, work);

			if (disp)
			{
				if (iter == 0)
				{
					std::cout << "iter: ";
					std::cout << "iflag: ";
					std::cout << "last: ";
					std::cout << "tot_iter: ";
					std::cout << "rej_iter: ";
					std::cout << "tt: ";
					std::cout << "tw: ";
					std::cout << "t: ";
					std::cout << "tout: ";
					std::cout << "err: ";
					std::cout << "hh: ";
					for (int i = 0; i < neqn; i++)
					{
						std::cout << "yw[" << i << "] ";
					}
					std::cout << '\n';
				}

				std::cout << std::setprecision(17);
				std::cout << iter << " ";
				std::cout << iflag << " ";
				std::cout << last << " ";
				std::cout << tot_iter << " ";
				std::cout << rej_iter << " ";
				std::cout << tt << " ";
				std::cout << tw << " ";
				std::cout << t << " ";
				std::cout << tout << " ";
				std::cout << err << " ";
				std::cout << hh << " ";
				for (int i = 0; i < neqn; i++)
				{
					std::cout << yw[i] << " ";
				}
				std::cout << '\n';
			}

			update_step_size(method.alpha, method.beta, method.safe, reject, err, errold, hh);

			// test if step was successful
			if (reject)
			{
				rej_iter++;
				continue;
			}

			// if step was successful and tout falls within the step prepare dense
			// output, interpolate, and return, if step was successful and tout falls
			// outside the step prepare the next step
			if (last || in_range(tout, tt, tw))
			{
				// iflag of 2 indicates that integration was successful
				iflag = 2;
				if (!method.fsal)
				{
					f(tw, yw, ywp, params);
				}
				method.dense(f, neqn, work, params);
				double temp = tt;
				tt = tw;
				tw = temp;
				for (int i = 0; i < neqn; i++)
				{
					yy[i] = yw[i];
					yyp[i] = ywp[i];
				}
				method.intrp(neqn, t, tout, y, yp, work);
				if (last)
				{
					initial_step_size(method.order, neqn, reltol, abstol, hh, yy, yyp);
				}
				return;
			}
			else
			{
				if (!method.fsal)
				{
					f(tw, yw, ywp, params);
				}
				tt = tw;
				for (int i = 0; i < neqn; i++)
				{
					yy[i] = yw[i];
					yyp[i] = ywp[i];
				}
			}
		}

		// iflag of 4 indicates that the maximum number of iterations was exceeded
		iflag = 4;
	}

	// Estimates the initial step size
	void initial_step_size(
		int order,
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

		hh = pow(err / neqn, -(1.0 / static_cast<double>(2 * order)));
	}

	// Estimates error and calculates next step size
	void update_step_size(
		double alpha,
		double beta,
		double safe,
		int& reject,
		double err,
		double& errold,
		double& hh)
	{
		// if err < 1.0 accept step, update hh, errold, and set reject to 0
		// if err > 1.0 reject step, update hh, and set reject to 1
		double scale;
		double min_scale = 0.333;
		double max_scale = 6.0;

		if (err < 1.0)
		{
			if (err == 0.0)
			{
				scale = max_scale;
			}
			else
			{
				scale = safe * pow(err, -alpha) * pow(errold, beta);
				scale = scale < min_scale ? min_scale : scale > max_scale ? max_scale : scale;
			}
			if (reject)
			{
				hh *= std::min(scale, 1.0);
			}
			else
			{
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
}