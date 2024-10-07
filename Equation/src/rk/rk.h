#pragma once

#include <vector>


namespace rk
{
	struct RKMethod
	{
		bool fsal;
		int order;
		int swork;
		int ywp_offset;
		double alpha;
		double beta;
		double safe;
		void (*dy)(void f(double t, double y[], double yp[], void* params), int neqn, std::vector<double>& work, void* params);
		double (*error)(int neqn, double reltol, double abstol, std::vector<double>& work);
		void (*dense)(void f(double t, double y[], double yp[], void* params), int neqn, std::vector<double>& work, void* params);
		void (*intrp)(int neqn, double& t, double tout, std::vector<double>& y, std::vector<double>& yp, std::vector<double>& work);
	};
	extern RKMethod method_rk32;
	extern RKMethod method_rk54;
	extern RKMethod method_rk853;
	bool in_range(double a, double r1, double r2);
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
		void* params);
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
		void* params);
	void initial_step_size(
		int order,
		int neqn,
		double reltol,
		double abstol,
		double& hh,
		double* yy,
		double* yyp);
	void update_step_size(
		double alpha,
		double beta,
		double safe,
		int& reject,
		double err,
		double& errold,
		double& hh);
}