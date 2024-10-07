#include <vector>

#pragma once

namespace rk853
{
	void dy(
		void f(double t, double y[], double yp[], void* params),
		int neqn,
		std::vector<double>& work,
		void* params);
	double error(
		int neqn,
		double reltol,
		double abstol,
		std::vector<double>& work);
	void dense(
		void f(double t, double y[], double yp[], void* params),
		int neqn,
		std::vector<double>& work,
		void* params);
	void intrp(
		int neqn,
		double& t,
		double tout,
		std::vector<double>& y,
		std::vector<double>& yp,
		std::vector<double>& work);
}