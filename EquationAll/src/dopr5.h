#include <vector>
#include <string>

namespace dopr5
{
	void init(
		void f(double t, double y[], double yp[], void* params),
		int& iflag,
		int neqn,
		double t,
		const std::vector<double>& y,
		std::vector<double>& yp,
		std::vector<int>& iwork,
		std::vector<double>& work,
		void* params);
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
		void f(double t, double y[], double yp[], void* params));
	void stepn(int neqn, double tout);
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
		void* params);

	void initial_step_size(int neqn, double& hh, double reltol, double abstol, double* yy, double* yyp);
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
		double* k7);
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
	);
	void intrp(
		double& t,
		double tout,
		std::vector<double>& y,
		std::vector<double>& yp,
		int neqn,
		double tt,
		double tw,
		double* r1,
		double* r2,
		double* r3,
		double* r4,
		double* r5
	);
	std::string get_error_string();
}