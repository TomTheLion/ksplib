#include <iostream>
#include <stdexcept>
#include <iomanip>
#include <vector>
#include <chrono>
#include "Equation.h"

void f(double t, double y[], double yp[], void* params)
{
	double r3 = pow(y[0] * y[0] + y[1] * y[1], 1.5);

	yp[0] = y[2];
	yp[1] = y[3];
	yp[2] = -y[0] / r3;
	yp[3] = -y[1] / r3;
}


int main()
{
	double pi = 3.14159265358979323846;
	std::vector<double> y = { 1.0, 0.0, 0.0, 1.0 };

	Equation eq2 = Equation(f, 0.0, y, "VOMS", 1e-8, 1e-8);
	Equation eq = eq2;
	//eq.step(2.0 * pi);

	double ss = 1.0;

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	eq.step(2.0 * pi);
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	double tout = eq.get_t();
	std::vector<double> yout = eq.get_y();

	double error = sqrt(pow(yout[0] - 1.0, 2.0) + pow(yout[1], 2));

	std::cout << std::setprecision(17) << tout << ", " << yout[0] << ", " << yout[1] << ", " << log10(error) << '\n';
	std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << '\n';

	for (int i = 0; i < 201; i++)
	{
		eq2.step(pi / 100.0 * i);
		double tout = eq2.get_t();
		std::vector<double> yout = eq2.get_y();
		std::vector<double> ypout = eq2.get_yp();
		std::cout << std::setprecision(17) << tout << ", " << yout[0] << ", " << yout[1] << ", " << ypout[0] << ", " << ypout[1] << '\n';
	}

	std::cout << eq.get_error_string();

	return 0;

}
int main2()
{
	try
	{
		double pi = 3.14159265358979323846;
		std::vector<double> y = { 1.0, 0.0, 0.0, 1.0 };

		for (int i = 0; i < 1000; i++)
		{
			double tol = pow(0.8 - 0.70 * i / 1000.0, 10);
			Equation eq = Equation(f, 0.0, y, "VOMS", tol, tol);
			Equation eq1 = eq;
			Equation eq2 = eq;
			Equation eq3 = eq;
			Equation eq4 = eq;

			std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

			eq.step(2.0 * pi);

			//double ss = 1.0;

			//for (int i = 0; i < 201; i++)
			//{
			//	eq.step(ss * pi / 100.0 * i);
			//	double tout = eq.get_t();
			//	std::vector<double> yout = eq.get_y();
			//	std::cout << std::setprecision(17) << tout << ", " << yout[0] << ", " << yout[1] << '\n';
			//}

			//// eq = Equation(f, eq.get_t(), eq.get_y(), "BOSH3", 1e-1, 1e-1);

			//for (int i = 200; i >= 0; i--)
			//{
			//	eq.step(ss * pi / 100.0 * i);
			//	double tout = eq.get_t();
			//	std::vector<double> yout = eq.get_y();
			//	std::cout << std::setprecision(17) << tout << ", " << yout[0] << ", " << yout[1] << '\n';
			//}
			//double tout = eq.get_t();
			//std::vector<double> yout = eq.get_y();

			std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

			double tout = eq.get_t();
			std::vector<double> yout = eq.get_y();
			double error = sqrt(pow(yout[0] - 1.0, 2.0) + pow(yout[1], 2));
			std::cout << std::setprecision(17) << tol << ", " << log10(error) << ", " << log10(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) << std::endl;
		}


	}
	catch (const std::exception& e)
	{
		std::cerr << "Error: " << e.what() << std::endl;
	}

	return 0;
}