#pragma once

#include <vector>
#include <string>


class Equation
{
public:
	Equation();
	Equation(
		void f(double t, double y[], double yp[], void* params),
		double t,
		const std::vector<double>& y,
		std::string method,
		double reltol,
		double abstol,
		void* params = nullptr);
	Equation(
		void f(double t, double y[], double yp[], void* params),
		int neqn,
		double t,
		const double* y,
		std::string method,
		double reltol,
		double abstol,
		void* params = nullptr);
	Equation(const Equation& equation);
	Equation& operator = (const Equation& equation);

	void step(double tout);
	void stepn(double tout);
	void stepn(double tout, double tlim);

	int get_tot_iter() const;
	int get_rej_iter() const;
	int get_iflag() const;
	int get_neqn() const;
	double get_t() const;
	double get_y(int i) const;
	std::vector<double> get_y() const;
	void get_y(int i, int n, double* x) const;
	double get_yp(int i) const;
	std::vector<double> get_yp() const;
	void get_yp(int i, int n, double* x) const;
	std::string get_error_string() const;

	void set_disp(bool disp);
private:
	enum class Method
	{
		NONE,
		RK32,
		RK54,
		RK853,
		VOMS
	};

	Method method_;

	bool disp_;
	int max_iter_;
	int tot_iter_;
	int rej_iter_;
	int iflag_;
	int neqn_;
	double reltol_;
	double abstol_;
	double t_;

	std::vector<double> y_;
	std::vector<double> yp_;
	std::vector<int> iwork_;
	std::vector<double> work_;

	void* params_;
	void(*f_)(double t, double y[], double yp[], void* params);
	void init_(std::string method);
	void step_(double tout, double tlim, bool lim);
};