#include "rk54.h"

namespace rk54
{
	// Node coefficients
	constexpr double c2 = 1.0 / 5.0;
	constexpr double c3 = 3.0 / 10.0;
	constexpr double c4 = 4.0 / 5.0;
	constexpr double c5 = 8.0 / 9.0;

	// Runge-Kutta matrix
	constexpr double a21 = 1.0 / 5.0;
	constexpr double a31 = 3.0 / 40.0;
	constexpr double a32 = 9.0 / 40.0;
	constexpr double a41 = 44.0 / 45.0;
	constexpr double a42 = -56.0 / 15.0;
	constexpr double a43 = 32.0 / 9.0;
	constexpr double a51 = 19372.0 / 6561.0;
	constexpr double a52 = -25360.0 / 2187.0;
	constexpr double a53 = 64448.0 / 6561.0;
	constexpr double a54 = -212.0 / 729.0;
	constexpr double a61 = 9017.0 / 3168.0;
	constexpr double a62 = -355.0 / 33.0;
	constexpr double a63 = 46732.0 / 5247.0;
	constexpr double a64 = 49.0 / 176.0;
	constexpr double a65 = -5103.0 / 18656.0;
	constexpr double a71 = 35.0 / 384.0;
	constexpr double a73 = 500.0 / 1113.0;
	constexpr double a74 = 125.0 / 192.0;
	constexpr double a75 = -2187.0 / 6784.0;
	constexpr double a76 = 11.0 / 84.0;

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

	void dy(
		void f(double t, double y[], double yp[], void* params),
		int neqn,
		std::vector<double>& work,
		void* params)
	{
		double& d = work[1];
		double& hh = work[2];
		double& tt = work[3];
		double* yy = &work[5];
		double* yyp = yy + neqn;
		double* yw = yyp + neqn;
		double* k2 = yw + neqn;
		double* k3 = k2 + neqn;
		double* k4 = k3 + neqn;
		double* k5 = k4 + neqn;
		double* k6 = k5 + neqn;
		double* k7 = k6 + neqn;

		double h = d * hh;

		for (int i = 0; i < neqn; i++)
			yw[i] = yy[i] + h * (a21 * yyp[i]);
		f(tt + c2 * h, yw, k2, params);

		for (int i = 0; i < neqn; i++)
			yw[i] = yy[i] + h * (a31 * yyp[i] + a32 * k2[i]);
		f(tt + c3 * h, yw, k3, params);

		for (int i = 0; i < neqn; i++)
			yw[i] = yy[i] + h * (a41 * yyp[i] + a42 * k2[i] + a43 * k3[i]);
		f(tt + c4 * h, yw, k4, params);

		for (int i = 0; i < neqn; i++)
			yw[i] = yy[i] + h * (a51 * yyp[i] + a52 * k2[i] + a53 * k3[i] + a54 * k4[i]);
		f(tt + c5 * h, yw, k5, params);

		for (int i = 0; i < neqn; i++)
			yw[i] = yy[i] + h * (a61 * yyp[i] + a62 * k2[i] + a63 * k3[i] + a64 * k4[i] + a65 * k5[i]);
		f(tt + h, yw, k6, params);

		for (int i = 0; i < neqn; i++)
			yw[i] = yy[i] + h * (a71 * yyp[i] + a73 * k3[i] + a74 * k4[i] + a75 * k5[i] + a76 * k6[i]);
		f(tt + h, yw, k7, params);
	}

	double error(
		int neqn,
		double reltol,
		double abstol,
		std::vector<double>& work
	)
	{
		double& hh = work[2];
		double* yy = &work[5];
		double* yyp = yy + neqn;
		double* yw = yyp + neqn;
		double* k2 = yw + neqn;
		double* k3 = k2 + neqn;
		double* k4 = k3 + neqn;
		double* k5 = k4 + neqn;
		double* k6 = k5 + neqn;
		double* k7 = k6 + neqn;

		double err = 0.0;
		for (int i = 0; i < neqn; i++)
		{
			double sci = abstol + reltol * std::max(abs(yy[i]), abs(yw[i]));
			double ei = hh * (e1 * yyp[i] + e3 * k3[i] + e4 * k4[i] + e5 * k5[i] + e6 * k6[i] + e7 * k7[i]);
			err += pow(ei / sci, 2.0);
		}

		return sqrt(err / neqn);
	}

	void dense(
		void f(double t, double y[], double yp[], void* params),
		int neqn,
		std::vector<double>& work,
		void* params)
	{
		double& tt = work[3];
		double& tw = work[4];
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

		double h = tw - tt;
		for (int i = 0; i < neqn; i++)
		{
			r1[i] = yy[i];
			r2[i] = yw[i] - yy[i];
			r3[i] = h * yyp[i] - r2[i];
			r4[i] = r2[i] - h * k7[i] - r3[i];
			r5[i] = h * (d1 * yyp[i] + d3 * k3[i] + d4 * k4[i] + d5 * k5[i] + d6 * k6[i] + d7 * k7[i]);
		}
	}

	void intrp(
		int neqn,
		double& t,
		double tout,
		std::vector<double>& y,
		std::vector<double>& yp,
		std::vector<double>& work)
	{
		double& tt = work[3];
		double& tw = work[4];
		double* r1 = &work[5 + 9 * neqn];
		double* r2 = r1 + neqn;
		double* r3 = r2 + neqn;
		double* r4 = r3 + neqn;
		double* r5 = r4 + neqn;

		t = tout;
		double h = tt - tw;
		double s = (tout - tw) / h;
		double s1 = 1.0 - s;

		for (int i = 0; i < neqn; i++)
		{
			double a4 = r5[i];
			double a3 = r4[i] + a4 * s1;
			double a2 = r3[i] + a3 * s;
			double a1 = r2[i] + a2 * s1;

			y[i] = r1[i] + s * a1;
			yp[i] = 1.0 / h * (a1 - s * (a2 - s1 * (a3 - s * a4)));
		}
	}
}