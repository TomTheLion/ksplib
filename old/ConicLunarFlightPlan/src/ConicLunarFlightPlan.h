#pragma once

#include <nlopt.hpp>
#include "astrodynamics.h"

class ConicLunarFlightPlan
{
public:

	ConicLunarFlightPlan(astrodynamics::ConicBody& planet, astrodynamics::ConicBody& moon);

	~ConicLunarFlightPlan();

    enum class TrajectoryMode
    {
        FREE_RETURN,
        LEAVE,
        RETURN
    };

	//
	// constraint functions
	//

    void set_mission(double initial_time, TrajectoryMode mode, double rp_planet, double rp_moon, double e_moon);
	void add_min_flight_time_constraint(double min_time);
	void add_max_flight_time_constraint(double max_time);
	void add_inclination_constraint(bool launch, double min, double max, Eigen::Vector3d n);
    void add_launch_plane_constraint(double tol, Eigen::Vector3d n);
	//
	// model functions
	//

    void init_model();
	void run_model(int max_eval, double eps, double eps_t, double eps_x);
    void set_conic_solution(std::vector<double> x);

    struct Result
    {
        int nlopt_code;
        int nlopt_num_evals;
        double nlopt_value;
        std::vector<double> nlopt_solution;
        std::vector<double> nlopt_constraints;
        std::vector<double> time;
        std::vector<int> leg;
        std::vector<std::vector<double>> r;
        std::vector<std::vector<double>> v;
        std::vector<std::vector<double>> rmoon;
        std::vector<std::vector<double>> vmoon;
    };

    Result output_result(double eps);

private:

    // 
    // variables
    //

    struct FlightPlanData
    {
        TrajectoryMode mode;
        double initial_time;
        double rp_planet;
        double rp_moon;
        double eps;
        double min_time;
        double max_time;
        double min_inclination_launch;
        double max_inclination_launch;    
        double n_launch_direction;
        Eigen::Vector3d n_launch;
        Eigen::Vector3d n_launch_plane;
        double min_inclination_arrival;
        double max_inclination_arrival;
        Eigen::Vector3d n_arrival;
        astrodynamics::ConicBody* planet;
        astrodynamics::ConicBody* moon;
        int count = 0;
    };

    nlopt::opt opt_;
    double e_moon_;

    FlightPlanData data_;
    std::vector<double> x_;

    std::tuple<std::vector<double>, std::vector<double>> bounds();

    static std::tuple<Eigen::Vector3d, Eigen::Vector3d> cartesian_state(const double*& x_ptr, double rho = 0);

    static void free_return_constraints(unsigned m, double* result, unsigned n, const double* x, double* grad, void* f_data);
    static void leave_constraints(unsigned m, double* result, unsigned n, const double* x, double* grad, void* f_data);
    static void return_constraints(unsigned m, double* result, unsigned n, const double* x, double* grad, void* f_data);
    static double free_return_objective(unsigned n, const double* x, double* grad, void* f_data);
    static double leave_objective(unsigned n, const double* x, double* grad, void* f_data);
    static double return_objective(unsigned n, const double* x, double* grad, void* f_data);

    static void objective_numerical_gradient(unsigned n, const double* x, double* grad,
        void* f_data, double(*func)(unsigned n, const double* x, double* grad, void* f_data));
    static void constraint_numerical_gradient(unsigned m, unsigned n, const double* x, double* grad, void* f_data,
        void(*func)(unsigned m, double* result, unsigned n, const double* x, double* grad, void* f_data));
};
