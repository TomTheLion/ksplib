#pragma once

class FlightPlan
{
public:

	FlightPlan(Ephemeris& ephemeris);

	~FlightPlan();

    //
    // flight sequence and basic constraint functions
    //

    void add_body(std::string body, Jdate jd, double radius_min, double d_scale, double orbits=0.0);
    void add_start_time_constraint(Jdate start_time);
    void add_min_flight_time_constraint(double min_time);
    void add_max_flight_time_constraint(double max_time);
    void add_max_c3_constraint(double c3);
    void add_arrival_eccentricity_constraint(double eccentricity);
    void add_inclination_constraint(bool launch, double min, double max);

    // 
    // patched conic model functions
    //
    
    void init_conic_model(double eps);     
    void set_conic_solution(std::vector<double> x);
    void run_conic_model(int max_eval, double eps, double eps_t, double eps_x);

    // 
    // nbody model functions
    //

    void init_nbody_model(double eps);
    void set_nbody_solution(std::vector<double> x);
    void run_nbody_model(int max_eval, double eps, double eps_t, double eps_f);

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
        std::vector<int> sequence;
        std::vector<std::string> bodies;
        std::vector<double> mu;
        std::vector<double> julian_time;
        std::vector<double> kerbal_time;
        std::vector<int> phase;
        std::vector<int> leg;
        std::vector<int> body;
        std::vector<std::vector<double>> dv;
        std::vector<std::vector<double>> r;
        std::vector<std::vector<double>> v;
        std::vector<std::vector<std::vector<double>>> rb;
        std::vector<std::vector<std::vector<double>>> vb;
    };

    // 
    // output functions
    //

    Result output_conic_result(double eps);
    Result output_nbody_result(double eps);

private:

    // 
    // variables
    //

    struct FlightPlanData
    {
        int num_bodies;
        int num_phases;
        int num_flybys;
        int num_moon_flybys;
        int index_home_body;
        int index_home_moon;
        double moon_mean_distance;
        double min_moon_distance_squared;
        double eps;
        double start_time;
        double min_time;
        double max_time;
        double max_c3;
        double eccentricity_arrival;
        double min_inclination_launch;
        double max_inclination_launch;
        double min_inclination_arrival;
        double max_inclination_arrival;
        double vinf_squared_launch;
        double vcirc_launch;
        double vinf_squared_arrival;
        double vell_arrival;
        std::vector<int> sequence;
        std::vector<double> radius_min;
        std::vector<double> radius_soi;
        std::vector<double> t_scale;
        std::vector<double> d_scale;
        std::vector<double> v_scale;
        Ephemeris* ephemeris;
    };

    struct IntegrationData
    {
        double ti;
        double tb;
        double tf;
        std::vector<double> xi;
        std::vector<double> yi;
        std::vector<double> yfb;
        std::vector<double> yff;
        std::vector<double> ypb;
        std::vector<double> ypf;
    };

    struct PhaseData
    {
        int b_forward;
        int b_backward;
        double d_scale_forward;
        double d_scale_backward;
        IntegrationData body_forward;
        IntegrationData moon_forward;
        IntegrationData soi_forward;
        IntegrationData dv1;
        IntegrationData dv3;
        IntegrationData soi_backward;
        IntegrationData moon_backward;
        IntegrationData body_backward;
    };

    nlopt::opt opt_a_;
    nlopt::opt opt_n_;
    std::vector<double> t_;
    std::vector<std::string> bodies_;
    std::vector<double> orbits_;
    std::string home_body_;
    std::string home_moon_;
    double day_;
    double year_;
    Eigen::Vector3d home_body_normal_;
    Eigen::Matrix3d n_transform_;

    FlightPlanData data_;
    std::vector<double> xa_;
    std::vector<double> xn_;

    // 
    // patched conic model functions
    //

    // set bounds for decisions variables for patched conic model
    std::tuple<std::vector<double>, std::vector<double>> bounds_conic_model();
    
    // calculate constraints and derivatives of nbody model for nlopt
    static void constraints_conic_model(unsigned m, double *result, unsigned n, const double* x, double* grad, void* f_data);
    
    // calculate objective function of nbody model for nlopt
    static double objective_conic_model(unsigned n, const double* x, double* grad, void* f_data);
   
    //
    // nbody model functions
    //

    // set bounds for decisions variables for nbody model
    std::tuple<std::vector<double>, std::vector<double>>  bounds_nbody_model();
    
    // calculate constraints and derivatives of nbody model for nlopt
    static void constraints_nbody_model(unsigned m, double *result, unsigned n, const double* x, double* grad, void* f_data);
    
    // calculate objective function of nbody model for nlopt
    static double objective_nbody_model(unsigned n, const double* x, double* grad, void* f_data);

    // converts polar input state vector to cartesian state vector
    static std::tuple<std::vector<double>, std::vector<double>> parse_input_state(const double*& x_ptr, double rho = 0);
    
    // returns phase data for given phase
    static void init_phase(PhaseData& phase_data, int i, const double*& t_ptr, const double*& x_ptr, FlightPlanData* data);

    // adds change in velocity to state vector
    static std::vector<double> add_dv(const double* yi, const double* dv, double dvn);

    // integrates trajectory
    static void integrate(IntegrationData& data, Ephemeris* ephemeris, bool forward, double d_scale, bool ref, int bref, double eps, double* grad);

    // returns the state transition matrix with respect to initial state in spherical coordinates, accounts for the scaling of the reference frame
    static Eigen::Matrix<double, 6, 6> spherical_stm(IntegrationData data, int direction, double d_scale, double v_scale);
    
    // returns the derivative of the dot product of r and v with respect to x
    static Eigen::Matrix<double, 6, 1> spherical_derivative_r_dot_v(std::vector<double> x);
};

