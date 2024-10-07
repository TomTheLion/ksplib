#include "wrapode.h"
#include "ode.hpp"

namespace wrap_ode
{
	// Initializes Equation
	// sets initial value of iflag, initializes workspace memory, copies initial
	// problem state, and calculates initial derivatives
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
		iflag = 1;
		yp.resize(neqn);
		iwork.resize(5);
		work.resize(100 + 21 * neqn);
		f(t, y.data(), yp.data(), params);
	}

	// Steps Equation from t to tout
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
		if (t == tout)
			return;

		// main integration loop
		for (int iter = 0; iter < max_iter; iter += 500)
		{
			tot_iter++;
			ode(f, neqn, &y[0], t, tout, reltol, abstol, iflag, &work[0], &iwork[0], params);
			if (abs(iflag) == 2)
				break;
			if (abs(iflag) == 3)
				rej_iter++;
		}

		if (rej_iter > 0)
			iflag = iflag > 0 ? 3 : -3;

		// copy derivatives to yp
		const double* yyp = &work[99 + neqn * 4];
		for (int i = 0; i < neqn; i++)
			yp[i] = yyp[i];
	}
}