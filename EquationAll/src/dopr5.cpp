#include <iostream>
#include "dopr5.h"

namespace dopr5
{
	constexpr double eps = 2.220446049250313E-016;

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

	bool in_range(double a, double r1, double r2)
	{
		return r1 <= r2 ? r1 <= a && a <= r2 : r2 <= a && a <= r1;
	}

	void init(
		void f(double t, double y[], double yp[], void* params),
		int& iflag,
		int neqn,
		double t,
		const std::vector<double>& y,
		std::vector<double>& yp,
		std::vector<int>& iwork,
		std::vector<double>& work,
		void* params)
	{
		iflag = 1;
		iwork.resize(1);
		yp.resize(neqn);
		work.resize(5 + 14 * neqn);
		work[0] = 1.0;
		work[3] = t;
		double* yy = &work[5];
		double* yyp = &work[5 + neqn];
		for (int i = 0; i < neqn; i++)
		{
			yy[i] = y[i];
		}
		f(t, yy, yyp, params);
		for (int i = 0; i < neqn; i++)
		{
			yp[i] = yyp[i];
		}
	}

	void step(
		int max_iter,
		int& iflag,
		int neqn,
		double reltol,
		double abstol,
		double& t,
		double tout,
		std::vector<int>& iwork,
		std::vector<double>& work,
		std::vector<double>& y,
		std::vector<double>& yp,
		void* params,
		void f(double t, double y[], double yp[], void* params))
	{
		int& reject = iwork[0];

		double& errold = work[0];
		double& d = work[1];
		double& hh = work[2];
		double& tt = work[3];
		double& tw = work[4];

		double tti = tt;

		const int iyy = 5;
		const int iyyp = iyy + neqn;
		const int iyw = iyyp + neqn;
		const int ik2 = iyw + neqn;
		const int ik3 = ik2 + neqn;
		const int ik4 = ik3 + neqn;
		const int ik5 = ik4 + neqn;
		const int ik6 = ik5 + neqn;
		const int ik7 = ik6 + neqn;
		const int ir1 = ik7 + neqn;
		const int ir2 = ir1 + neqn;
		const int ir3 = ir2 + neqn;
		const int ir4 = ir3 + neqn;
		const int ir5 = ir4 + neqn;

		double* pwork = work.data();;

		if (iflag == 1)
		{
			initial_step_size(neqn, hh, reltol, abstol, pwork + iyy, pwork + iyyp);
		}
		else if (iflag == 2 && in_range(tout, tt, tw))
		{
			intrp(t, tout, y, yp, neqn, tt, d * hh, pwork + ir1, pwork + ir2, pwork + ir3, pwork + ir4, pwork + ir5);
			// std::cout << "intrp\n";
			return;
		}

		d = tout > tt ? 1.0 : -1.0;
		for (int iter = 0; iter < max_iter; iter++)
		{
			if (hh < 4.0 * eps * abs(tt - tti))
			{
				throw std::runtime_error("Equation failed, minimum step size reached.");
			}

			dy(f, neqn, tt, d * hh, pwork + iyy, pwork + iyyp, pwork + iyw, pwork + ik2, pwork + ik3, pwork + ik4, pwork + ik5, pwork + ik6, pwork + ik7, params);
			update_step_size(neqn, reltol, abstol, reject, errold, d, hh, tt, tw, pwork + iyy, pwork + iyyp, pwork + iyw, pwork + ik3, pwork + ik4, pwork + ik5, pwork + ik6, pwork + ik7);
				
			if (reject)
				continue;

			if (in_range(tout, tt, tw))
			{
				dense(neqn, d * hh, pwork + iyy, pwork + iyyp, pwork + iyw, pwork + ik3, pwork + ik4, pwork + ik5, pwork + ik6, pwork + ik7, pwork + ir1, pwork + ir2, pwork + ir3, pwork + ir4, pwork + ir5);
				iflag = 2;
				double temp = tt;
				tt = tw;
				tw = tt;
				for (int i = 0; i < neqn; i++)
				{
					work[iyy + i] = work[iyw + i];
					work[iyyp + i] = work[ik7 + i];
				}
				intrp(t, tout, y, yp, neqn, tt, d * hh, pwork + ir1, pwork + ir2, pwork + ir3, pwork + ir4, pwork + ir5);
				// std::cout << "step\n";
				return;
			}
			else
			{
				tt = tw;
				for (int i = 0; i < neqn; i++)
				{
					work[iyy + i] = work[iyw + i];
					work[iyyp + i] = work[ik7 + i];
				}
			}
		}

		iflag = 4;
		throw std::runtime_error("Equation failed, maximum iterations reached.");
	}

	void stepn(int neqn, double tout)
	{

	}

	void dy(
		void f(double t, double y[], double yp[], void* params),
		int neqn,
		double tt,
		double h,
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
			yw[i] = yy[i] + h * (b11 * yyp[i]);
		f(tt + a1 * h, yw, k2, params);

		for (int i = 0; i < neqn; i++)
			yw[i] = yy[i] + h * (b21 * yyp[i] + b22 * k2[i]);
		f(tt + a2 * h, yw, k3, params);

		for (int i = 0; i < neqn; i++)
			yw[i] = yy[i] + h * (b31 * yyp[i] + b32 * k2[i] + b33 * k3[i]);
		f(tt + a3 * h, yw, k4, params);

		for (int i = 0; i < neqn; i++)
			yw[i] = yy[i] + h * (b41 * yyp[i] + b42 * k2[i] + b43 * k3[i] + b44 * k4[i]);
		f(tt + a4 * h, yw, k5, params);

		for (int i = 0; i < neqn; i++)
			yw[i] = yy[i] + h * (b51 * yyp[i] + b52 * k2[i] + b53 * k3[i] + b54 * k4[i] + b55 * k5[i]);
		f(tt + h, yw, k6, params);

		for (int i = 0; i < neqn; i++)
			yw[i] = yy[i] + h * (b61 * yyp[i] + b63 * k3[i] + b64 * k4[i] + b65 * k5[i] + b66 * k6[i]);
		f(tt + h, yw, k7, params);
	}

	void initial_step_size(int neqn, double& hh, double reltol, double abstol, double* yy, double* yyp)
	{
		double err = 0.0;

		for (int i = 0; i < neqn; i++)
		{
			double sci = (1.0 + reltol / abstol * abs(yy[i]));
			err = std::max(err, abs(yyp[i]) / sci);
		}

		hh = pow(abstol / err, 1.0 / 6.0);
	}

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
		const double alpha = 0.17;
		const double beta = 0.03;
		const double safe = 0.92;
		const double min_scale = 0.2;
		const double max_scale = 10.0;

		double scale;
		double err = 0.0;

		for (int i = 0; i < neqn; i++)
		{
			double sci = abstol + reltol * std::max(abs(yy[i]), abs(yw[i]));
			double ei = hh * (e1 * yyp[i] + e3 * k3[i] + e4 * k4[i] + e5 * k5[i] + e6 * k6[i] + e7 * k7[i]);
			err += pow(ei / sci, 2.0);
		}

		err = sqrt(err / neqn);

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
			reject = false;
		}
		else
		{
			scale = std::max(safe * pow(err, -alpha), min_scale);
			hh *= scale;
			reject = true;
		}
	}

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
		double* r5
	)
	{
		for (int i = 0; i < neqn; i++)
		{
			r1[i] = yy[i];
			double dy = yw[i] - yy[i];
			double bspl = h * yyp[i] - dy;
			r2[i] = dy;
			r3[i] = bspl;
			r4[i] = dy - h * k7[i] - bspl;
			r5[i] = h * (
				d1 * yyp[i]
				+ d3 * k3[i]
				+ d4 * k4[i]
				+ d5 * k5[i]
				+ d6 * k6[i]
				+ d7 * k7[i]);
		}
	}

	void intrp(
		double& t,
		double tout,
		std::vector<double>& y,
		std::vector<double>& yp,
		int neqn,
		double tt,
		double h,
		double* r1,
		double* r2,
		double* r3,
		double* r4,
		double* r5
	)
	{
		t = tout;
		double s = (tout - tt) / h + 1.0;
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

	std::string get_error_string()
	{
		return std::string();
	}
}