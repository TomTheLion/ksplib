#include <iostream>
#include <iomanip>
#include <vector>
#include "Equation.h"

void f(double t, double y[], double yp[], void* params)
{
	double r3 = pow(y[0] * y[0] + y[1] * y[1], 1.5);

	yp[0] = y[2];
	yp[1] = y[3];
	yp[2] = -y[0] / r3;
	yp[3] = -y[1] / r3;

}


//
// potentially rethink dense output
//

int main()
{
	double pi = 3.14159265358979323846;
	std::vector<double> y = { 1.0, 0.0, 0.0, 1.0 };
	Equation eq = Equation(f, 0.0, y, "ADB2");

	for (int i = 0; i < 201; i++)
	{
		eq.step(pi / 100.0 * i);
		double tout = eq.get_t();
		std::vector<double> yout = eq.get_y();

		std::cout << std::fixed << std::setprecision(6) << std::setw(9) << tout << ", " << (yout[0] < 0 ? "" : " ") << std::setw(8) << yout[0] << ", " << (yout[1] < 0 ? "" : " ") << std::setw(8) << yout[1] << '\n';
	}

	return 0;
}