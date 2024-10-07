#include <iostream>
#include <iomanip>
#include "Equation.h"
#include <chrono>

void f(double t, double y[], double yp[], void* params)
{
	double mag = pow(y[0] * y[0] + y[1] * y[1] + y[2] * y[2], -1.5);
	yp[0] = y[3];
	yp[1] = y[4];
	yp[2] = y[5];
	yp[3] = -y[0] * mag;
	yp[4] = -y[1] * mag;
	yp[5] = -y[2] * mag;
}

int main()
{
	double pi = 3.14159265358979323846;
	double e = 0.0;
	double semi_major_axis = 1.0 / (1.0 + e);
	double period = 2.0 * pi * pow(semi_major_axis, 1.5);
	double velocity = sqrt((1.0 - e) / (1.0 + e) / semi_major_axis);
	std::vector<double> y = { 1.0, 0.0, 0.0, 0.0, velocity, 0.0 };

	Equation eq1 = Equation();
	Equation eq2 = Equation(f, 0.0, y, "DOPR5", 1e-10, 1e-10);

	try
	{
		double dt = pi / 100.0;
		for (int i = 0; i < 201; i++)
		{
			double t = dt * i;
			eq2.step(t);
			double tout = eq2.get_t();
			std::vector<double> yout = eq2.get_y();
			std::cout << std::setprecision(17) << tout << ", " << yout[0] << ", " << yout[1] << ", " << yout[2] << '\n';

		}

	}
	catch (std::runtime_error& e) {
		std::cerr << "Error: " << e.what() << std::endl;
	}
	


	//std::vector<double> a{ 0.0, 1.0, 2.5 };
	//double* a_ptr = a.data();
	//std::vector<double> b = std::vector<double>(a_ptr + 1, a_ptr + 3);

	//double pi = 3.14159265358979323846;
	//
	//std::vector<double> es{ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999 };
	//std::vector<double> tols{ 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-12, 1e-13, 1e-14 };
	//std::vector<double> yout(6);

	//double e = 0.0;

	//double semi_major_axis = 1.0 / (1.0 + e);
	//double period = 2.0 * pi * pow(semi_major_axis, 1.5);
	//double velocity = sqrt((1.0 - e) / (1.0 + e) / semi_major_axis);
	//std::vector<double> y = { 1.0, 0.0, 0.0, 0.0, velocity, 0.0 };

	//std::vector<double> dt(101, 0.0);

	//for (int j = 0; j < 1; j++)
	//{
	//	for (int i = 0; i < 1; i++)
	//	{
	//		double tol = pow(10.0, -(2.0 + 12.0 * i / 100.0));

	//		tol = 1e-14;

	//		Equation eq = Equation(f, 6, y.data(), 0.0, tol, tol, nullptr);
	//		
	//		auto start = std::chrono::high_resolution_clock::now();
	//		eq.step(10.0 * period);

	//		auto end = std::chrono::high_resolution_clock::now();
	//		std::chrono::duration<double, std::micro> elapsed = end - start;

	//		eq.get_y(0, 6, yout.data());

	//		dt[i] += elapsed.count();
	//	}
	//}
	//std::cout << "boace" << '\n';
	//for (int i = 0; i < 1; i++)
	//{
	//	double tol = pow(10.0, -(2.0 + 12.0 * i / 100.0));

	//	tol = 1e-7;

	//	Equation eq = Equation(f, 6, y.data(), 0.0, tol, tol, nullptr);
	//	eq.step(period);
	//	eq.get_y(0, 6, yout.data());

	//	std::cout << std::setprecision(17) << dt[i] / 1000.0 << ", ";
	//	std::cout << std::setprecision(17) << e << ", ";
	//	std::cout << std::setprecision(17) << tol << ", ";

	//	for (int i = 0; i < 6; i++)
	//	{
	//		std::cout << std::setprecision(17) << yout[i] - y[i] << ", ";
	//	}

	//	std::cout << '\n';
	//}







	return 0;
}