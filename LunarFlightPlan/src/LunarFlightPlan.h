#pragma once

class LunarFlightPlan
{
public:

	LunarFlightPlan(Ephemeris& ephemeris);

	~LunarFlightPlan();

	//
	// constraint functions
	//

    void set_mission(Jdate initial_time, bool free_return, double rp_earth, double rp_moon, double* initial_orbit = nullptr);
	void add_min_flight_time_constraint(double min_time);
	void add_max_flight_time_constraint(double max_time);
	void add_inclination_constraint(bool launch, double min, double max, Eigen::Vector3d n);

	//
	// model functions
	//

    void init_model();
	void run_model(int max_eval, double eps, double eps_t, double eps_f);

    struct Result
    {
        int nlopt_code;
        int nlopt_num_evals;
        double nlopt_value;
        double time_scale;
        double distance_scale;
        double velocity_scale;
        std::vector<double> nlopt_solution;
        std::vector<double> nlopt_constraints;
        std::vector<double> mu;
        std::vector<double> julian_time;
        std::vector<double> kerbal_time;
        std::vector<int> leg;
        std::vector<std::vector<double>> r;
        std::vector<std::vector<double>> v;
        std::vector<std::vector<double>> rsun;
        std::vector<std::vector<double>> vsun;
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
        bool initial_orbit;
        bool free_return;
        double initial_time;
        Eigen::Vector3d initial_position;
        Eigen::Vector3d initial_velocity;
        double rp_earth;
        double rp_moon;
        double eps;
        double min_time;
        double max_time;
        double min_inclination_launch;
        double max_inclination_launch;    
        Eigen::Vector3d n_launch;
        double min_inclination_arrival;
        double max_inclination_arrival;
        Eigen::Vector3d n_arrival;
        Ephemeris* ephemeris;
    };

    nlopt::opt opt_;
    double day_;
    double year_;

    FlightPlanData data_;
    std::vector<double> x_;

    std::tuple<std::vector<double>, std::vector<double>> bounds();

    static void constraints(unsigned m, double* result, unsigned n, const double* x, double* grad, void* f_data);
    static double objective(unsigned n, const double* x, double* grad, void* f_data);
};
