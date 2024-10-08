#include <iostream>
#include <stdexcept>
#include <iomanip>
#include <vector>
#include <chrono>
#include "Equation.h"
#include "Spl.h"
#include "Scalar.h"
#include <Eigen/Dense>

void f(double t, double y[], double yp[], void* params)
{
	double r3 = pow(y[0] * y[0] + y[1] * y[1], 1.5);

	yp[0] = y[2];
	yp[1] = y[3];
	yp[2] = -y[0] / r3;
	yp[3] = -y[1] / r3;
}

static void yp_vac_stm(double t, double y[], double yp[], void* params)
{
	// state
	Eigen::Map<Eigen::Vector3d> r(y);
	Eigen::Map<Eigen::Vector3d> v(y + 3);
	Eigen::Map<Eigen::Vector3d> l(y + 6);
	Eigen::Map<Eigen::Vector3d> ld(y + 9);
	double m = y[12];
	double ft = y[13];
	double ve = y[14];
	Eigen::Map<Eigen::Matrix<double, 13, 13>> stm(y + 15);

	// derivatives
	Eigen::Map<Eigen::Vector3d> dr(yp);
	Eigen::Map<Eigen::Vector3d> dv(yp + 3);
	Eigen::Map<Eigen::Vector3d> dl(yp + 6);
	Eigen::Map<Eigen::Vector3d> dld(yp + 9);
	double dm = yp[12];
	double dft = yp[13];
	double dve = yp[14];
	Eigen::Map<Eigen::Matrix<double, 13, 13>> dstm(yp + 15);
	Eigen::Ref<Eigen::Matrix3d> dr_dv = dstm.block<3, 3>(0, 3);
	Eigen::Ref<Eigen::Matrix3d> dv_dr = dstm.block<3, 3>(3, 0);
	Eigen::Ref<Eigen::Matrix3d> dv_dl = dstm.block<3, 3>(3, 6);
	Eigen::Ref<Eigen::Vector3d> dv_dm = dstm.block<3, 1>(3, 12);

	Eigen::Ref<Eigen::Matrix3d> dl_dld = dstm.block<3, 3>(6, 9);
	Eigen::Ref<Eigen::Matrix3d> dld_dl = dstm.block<3, 3>(9, 6);

	double& ddmdm = yp[183];

	// is it faster to repeat multiplcations?

	// set derivatives
	dr = v;
	dv = -r / pow(r.norm(), 3.0) + l.normalized() * ft / m;
	dl = ld;
	dld = -l;
	dm = -ft / ve;
	dft = 0.0;
	dve = 0.0;

	// set stm derivatives
	dstm.setZero();
	dr_dv.setIdentity();
	dv_dr(0, 0) = 3.0 * r(0) * r(0) - r.squaredNorm();
	dv_dr(0, 1) = 3.0 * r(0) * r(1);
	dv_dr(0, 2) = 3.0 * r(0) * r(2);

	dv_dr(1, 0) = 3.0 * r(1) * r(0);
	dv_dr(1, 1) = 3.0 * r(1) * r(1) - r.squaredNorm();
	dv_dr(1, 2) = 3.0 * r(1) * r(2);

	dv_dr(2, 0) = 3.0 * r(2) * r(0);
	dv_dr(2, 1) = 3.0 * r(2) * r(1);
	dv_dr(2, 2) = 3.0 * r(2) * r(2) - r.squaredNorm();

	dv_dr *= pow(r.norm(), -5.0);

	dv_dl(0, 0) = (l(1) * l(1) + l(2) * l(2)) * pow(l.norm(), -3.0);
	dv_dl(0, 1) = -l(0) * l(1) * pow(l.norm(), -3.0);
	dv_dl(0, 2) = -l(0) * l(2) * pow(l.norm(), -3.0);

	dv_dl(1, 0) = -l(1) * l(0) * pow(l.norm(), -3.0);
	dv_dl(1, 1) = (l(0) * l(0) + l(2) * l(2)) * pow(l.norm(), -3.0);
	dv_dl(1, 2) = -l(1) * l(2) * pow(l.norm(), -3.0);

	dv_dl(2, 0) = -l(2) * l(0) * pow(l.norm(), -3.0);
	dv_dl(2, 1) = -l(2) * l(1) * pow(l.norm(), -3.0);
	dv_dl(2, 2) = (l(0) * l(0) + l(1) * l(1)) * pow(l.norm(), -3.0);

	dv_dl *= ft / m;

	dl_dld.setIdentity();
	dld_dl.setIdentity();
	dld_dl = -dld_dl;

	dv_dm = -l.normalized() * ft / m / m;
}

void print_y(double* y)
{
	for (size_t i = 0; i < 15; i++)
	{
		std::cout << std::setprecision(17) << y[i] << " ";
	}
	std::cout << '\n';
	for (size_t i = 0; i < 13; i++)
	{
		for (size_t j = 0; j < 13; j++)
		{
			std::cout << std::setprecision(17) << y[i  + j * 13 + 15] << " ";
		}
		std::cout << '\n';
	}
}

void print_dyp_dy(double* y)
{
	std::vector<double> yp1_(184);
	std::vector<double> yp2_(184);
	for (size_t i = 0; i < 13; i++)
	{
		std::vector<double> y_ = std::vector<double>(y, y + 184);
		yp_vac_stm(0.0, y_.data(), yp1_.data(), nullptr);
		y_[i] += 1e-10;
		yp_vac_stm(0.0, y_.data(), yp2_.data(), nullptr);
		for (size_t j = 0; j < 13; j++)
		{
			std::cout << (yp2_[j] - yp1_[j]) / 1e-10 << " ";
		}
		std::cout << '\n';
	}
}

int main()
{
	double pi = 3.14159265358979323846;
	std::vector<double> y(184);
	std::vector<double> yp(184);
	Eigen::Map<Eigen::Vector3d> r(y.data());
	Eigen::Map<Eigen::Vector3d> v(y.data() + 3);
	Eigen::Map<Eigen::Vector3d> l(y.data() + 6);
	Eigen::Map<Eigen::Vector3d> ld(y.data() + 9);
	double& m = y[12];
	double& ft = y[13];
	double& ve = y[14];
	Eigen::Map<Eigen::Matrix<double, 13, 13>> stm(y.data() + 15);
	r << 1.01, 0.02, 0.03;
	v << 0.04, 0.85, 0.06;
	l << 0.37, 0.88, 0.09;
	ld << -1.10, -0.11, -0.12;
	m = 1.44;
	ft = 1.33;
	ve = 1.11;
	stm.setIdentity();

	yp_vac_stm(0.0, y.data(), yp.data(), nullptr);

	//print_dyp_dy(y.data());

	print_y(yp.data());

	//Equation eq54 = Equation(f, 0.0, y, "RK54", 1e-3, 1e-3);

	return 0;

}

int main2()
{
	Scalar qs = Scalar(300.0, 600000.0, 10000.0);

	double time = 60.0;
	double distance = 700000.0;
	double velocity = 2000.0;
	double acceleration = 30.0;
	double mass = 13000.0;
	double mass_rate = 30.0;
	double force = 175000.0;

	double scaled_time = qs.ndim(Scalar::Quantity::FORCE, force);

	std::cout << scaled_time << '\n';

	double rescaled_time = qs.rdim(Scalar::Quantity::FORCE, scaled_time);

	std::cout << rescaled_time << '\n';
	
	
	//int nk = 3;
	//std::vector<double> tk = { 0, 0, 0, 0, 0.202020202020202, 0.303030303030303, 0.404040404040404, 0.505050505050505, 0.606060606060606, 0.707070707070707, 0.808080808080808, 0.909090909090909, 1.01010101010101, 1.11111111111111, 1.21212121212121, 1.31313131313131, 1.41414141414141, 1.51515151515151, 1.61616161616161, 1.71717171717171, 1.81818181818181, 1.91919191919191, 2.02020202020202, 2.12121212121212, 2.22222222222222, 2.32323232323232, 2.42424242424242, 2.52525252525252, 2.62626262626262, 2.72727272727272, 2.82828282828282, 2.92929292929292, 3.03030303030303, 3.13131313131313, 3.23232323232323, 3.33333333333333, 3.43434343434343, 3.53535353535353, 3.63636363636363, 3.73737373737373, 3.83838383838383, 3.93939393939393, 4.04040404040404, 4.14141414141414, 4.24242424242424, 4.34343434343434, 4.44444444444444, 4.54545454545454, 4.64646464646464, 4.74747474747474, 4.84848484848484, 4.94949494949495, 5.05050505050505, 5.15151515151515, 5.25252525252525, 5.35353535353535, 5.45454545454545, 5.55555555555555, 5.65656565656565, 5.75757575757575, 5.85858585858585, 5.95959595959595, 6.06060606060606, 6.16161616161616, 6.26262626262626, 6.36363636363636, 6.46464646464646, 6.56565656565656, 6.66666666666666, 6.76767676767676, 6.86868686868686, 6.96969696969697, 7.07070707070707, 7.17171717171717, 7.27272727272727, 7.37373737373737, 7.47474747474747, 7.57575757575757, 7.67676767676767, 7.77777777777777, 7.87878787878787, 7.97979797979797, 8.08080808080808, 8.18181818181818, 8.28282828282828, 8.38383838383838, 8.48484848484848, 8.58585858585858, 8.68686868686868, 8.78787878787878, 8.88888888888889, 8.98989898989899, 9.09090909090909, 9.19191919191919, 9.29292929292929, 9.39393939393939, 9.49494949494949, 9.59595959595959, 9.69696969696969, 9.79797979797979, 10, 10, 10, 10 };
	//std::vector<double> ck = { -8.60663863713809E-19, 0.0673412849543499, 0.168346682862375, 0.298921669812646, 0.393805717835317, 0.484675131736063, 0.570603597848035, 0.650715118325587, 0.724193009883853, 0.790288211015763, 0.848326923118636, 0.897717478127587, 0.937956370550751, 0.968633390295112, 0.989435804524596, 1.000151545767, 1.00067137380972, 0.990989989334231, 0.971206087939388, 0.941521354002156, 0.902238404632772, 0.853757704684484, 0.796573484267341, 0.731268700384445, 0.658509094053556, 0.579036403497982, 0.493660802593952, 0.403252641659749, 0.308733574784027, 0.2110671641446, 0.111249057100728, 0.0102968361972644, -0.090760354446579, -0.190892300611394, -0.289078220360752, -0.384317170268191, -0.475638249403154, -0.562110497052629, -0.642852383277535, -0.717040795553557, -0.783919429883376, -0.842806500838148, -0.893101691929253, -0.93429227545568, -0.965958339439028, -0.987777068360875, -0.999526034063049, -1.0010854632622, -0.992439458562757, -0.97367616052083, -0.944986849106923, -0.906663993727474, -0.859098271684002, -0.802774585464826, -0.738267119470509, -0.666233486566614, -0.587408024135967, -0.502594307972927, -0.412656960335793, -0.318512835669074, -0.221121673851619, -0.121476316254828, -0.0205925843520685, 0.0805010759395442, 0.180774078615582, 0.27920420375759, 0.374788018419398, 0.466551105992132, 0.553557999765983, 0.63492171942237, 0.709812813237969, 0.777467813821059, 0.83719702117928, 0.888391533775374, 0.930529455893695, 0.96318121803735, 0.986013956117846, 0.998794904794207, 1.00139377036857, 0.993784059048161, 0.976043347032716, 0.948352489674189, 0.910993777770655, 0.86434805978996, 0.808890859360203, 0.745187527606597, 0.673887479754473, 0.595717574749095, 0.511474705396124, 0.422017674512108, 0.328258440087084, 0.231152818020548, 0.131690739781032, 0.030886154713067, -0.0702332842664515, -0.170636780051141, -0.269300589497563, -0.397192626728684, -0.487523700766889, -0.544021110889369, 0, 0, 0, 0 };
	//std::vector<double> rk(100000);

	//Spl s(tk.data(), ck.data(), tk.size(), 3);

	//std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	//for (size_t i = 0; i < 100000; i++)
	//{
	//	double x = static_cast<double>(i) / 10000;
	//	rk[i] = s.eval(x);
	//	// std::cout << x << " " << rk[i] << '\n';
	//}
	//std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	//std::cout << std::setprecision(17) << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << '\n';

	//double pi = 3.14159265358979323846;
	//std::vector<double> y = { 1.0, 0.0, 0.0, 1.0 };
	//std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	//Equation eq = Equation(f, 0.0, y, "RK32", 1e-12, 1e-12, nullptr);
	//std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	//std::cout << std::setprecision(17) << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << '\n';


	//double tout = 0.0;

	//for (size_t i = 0; i < 100; i++)
	//{

	//		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	//		tout += 2.0 * pi;
	//		eq.step(tout);
	//		double y0 = eq.get_y(0);
	//		double y1 = eq.get_y(1);
	//		double y2 = eq.get_y(2);
	//		double y3 = eq.get_y(3);
	//		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	//		std::cout << std::setprecision(17) << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << " " << y0 << " " << y1 << " " << y2 << " " << y3 << " " << '\n';

	//}

	return 0;

}

void outputeq(Equation& eq)
{
	std::cout << std::setprecision(17);
	for (size_t i = 0; i < eq.get_neqn(); i++)
		std::cout << eq.get_y(i) << " ";
	std::cout << '\n';
}

void testeq(Equation& eq)
{
	double pi = 3.14159265358979323846;
	eq.step(0.25); outputeq(eq);
	eq.stepn(0.50, 0.50); outputeq(eq);
	eq.stepn(0.45, 0.35); outputeq(eq);
	eq.step(pi); outputeq(eq);
	eq.step(-pi); outputeq(eq);
	eq.step(2.0 * pi); outputeq(eq);
	eq.step(0.0); outputeq(eq);
	eq.stepn(4.0 * pi, 4.0 * pi); outputeq(eq);
	std::cout << '\n';
}



//int main2()
//{
//	try
//	{
//		double pi = 3.14159265358979323846;
//		std::vector<double> y = { 1.0, 0.0, 0.0, 1.0 };
//
//		for (int i = 0; i < 1000; i++)
//		{
//			double tol = pow(0.8 - 0.70 * i / 1000.0, 10);
//			Equation eq = Equation(f, 0.0, y, "VOMS", tol, tol);
//			Equation eq1 = eq;
//			Equation eq2 = eq;
//			Equation eq3 = eq;
//			Equation eq4 = eq;
//
//			std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
//
//			eq.step(2.0 * pi);
//
//			//double ss = 1.0;
//
//			//for (int i = 0; i < 201; i++)
//			//{
//			//	eq.step(ss * pi / 100.0 * i);
//			//	double tout = eq.get_t();
//			//	std::vector<double> yout = eq.get_y();
//			//	std::cout << std::setprecision(17) << tout << ", " << yout[0] << ", " << yout[1] << '\n';
//			//}
//
//			//// eq = Equation(f, eq.get_t(), eq.get_y(), "BOSH3", 1e-1, 1e-1);
//
//			//for (int i = 200; i >= 0; i--)
//			//{
//			//	eq.step(ss * pi / 100.0 * i);
//			//	double tout = eq.get_t();
//			//	std::vector<double> yout = eq.get_y();
//			//	std::cout << std::setprecision(17) << tout << ", " << yout[0] << ", " << yout[1] << '\n';
//			//}
//			//double tout = eq.get_t();
//			//std::vector<double> yout = eq.get_y();
//
//			std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
//
//			double tout = eq.get_t();
//			std::vector<double> yout = eq.get_y();
//			double error = sqrt(pow(yout[0] - 1.0, 2.0) + pow(yout[1], 2));
//			std::cout << std::setprecision(17) << tol << ", " << log10(error) << ", " << log10(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) << std::endl;
//		}
//
//
//	}
//	catch (const std::exception& e)
//	{
//		std::cerr << "Error: " << e.what() << std::endl;
//	}
//
//	return 0;
//}