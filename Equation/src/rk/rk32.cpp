#include "rk32.h"

namespace rk32
{
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

		double h = d * hh;

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

		double err = 0.0;
		for (int i = 0; i < neqn; i++)
		{
			double sci = abstol + reltol * std::max(abs(yy[i]), abs(yw[i]));
			double ei = hh * (e1 * yyp[i] + e2 * k2[i] + e3 * k3[i] + e4 * k4[i]);
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
		double* k4 = &work[5 + 5 * neqn];
		double* r1 = k4 + neqn;
		double* r2 = r1 + neqn;
		double* r3 = r2 + neqn;
		double* r4 = r3 + neqn;

		double h = tw - tt;
		for (int i = 0; i < neqn; i++)
		{
			r1[i] = yy[i];
			r2[i] = yw[i] - yy[i];
			r3[i] = h * yyp[i] - r2[i];
			r4[i] = r2[i] - h * k4[i] - r3[i];
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
		double* r1 = &work[5 + 6 * neqn];
		double* r2 = r1 + neqn;
		double* r3 = r2 + neqn;
		double* r4 = r3 + neqn;

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