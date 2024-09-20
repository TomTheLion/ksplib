#include "wrapode.h"
#include "ode.hpp"

namespace wrap_ode
{
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
		iwork.resize(5);
		work.resize(100 + 21 * neqn);
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
		if (t == tout)
			return;

		// main integration loop
		for (int iter = 0; iter < max_iter; iter += 500)
		{
			ode(f, neqn, &y[0], t, tout, reltol, abstol, iflag, &work[0], &iwork[0], params);
			if (iflag == 2)
				break;
		}

		// copy derivatives to yp
		const double* yyp = &work[99 + neqn * 4];
		for (int i = 0; i < neqn; i++)
			yp[i] = yyp[i];
	}
}