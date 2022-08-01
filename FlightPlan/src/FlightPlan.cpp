#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <nlopt.hpp>

#include "Jdate.h"
#include "Equation.h"
#include "Ephemeris.h"
#include "astrodynamics.h"
#include "FlightPlan.h"

// create an initialized FlightPlan object
// references ephemeris from astrodynamics namespace
// sets default variables
FlightPlan::FlightPlan(Ephemeris& ephemeris)
{
    data_.ephemeris = &ephemeris;

    // initialize constraints with default values
    data_.start_time = -1.0;
    data_.min_start_time = -1.0;
    data_.min_time = -1.0;
    data_.max_time = data_.ephemeris->get_max_time();
    data_.max_c3 = -1.0;
    data_.eccentricity_arrival = -1.0;
    data_.min_inclination_launch = 0.0;
    data_.max_inclination_launch = astrodynamics::pi;
    data_.min_inclination_arrival = 0.0;
    data_.max_inclination_arrival = astrodynamics::pi;

    // set solar system specific parameters
    home_body_ = "Earth";
    home_moon_ = "Moon";
    day_ = 86400.0 / data_.ephemeris->get_time_scale();
    year_ = day_ * 365.25;
    n_transform_.row(0) = data_.ephemeris->get_position("Earth", 0.0).normalized();
    n_transform_.row(1) = data_.ephemeris->get_velocity("Earth", 0.0).normalized();
    n_transform_.row(2) = n_transform_.row(0).cross(n_transform_.row(1)).normalized();
    n_transform_.row(1) = n_transform_.row(2).cross(n_transform_.row(0)).normalized();
    home_body_normal_ = n_transform_.row(2);

    data_.index_home_body = data_.ephemeris->get_index_of_body("Earth");
    data_.index_home_moon = data_.ephemeris->get_index_of_body("Moon");
    data_.moon_mean_distance = 385000000.0 / data_.ephemeris->get_distance_scale();
    data_.min_moon_distance_squared = 2.5e15 / pow(data_.ephemeris->get_distance_scale(), 2.0);
};

// destroy FlightPlan object
FlightPlan::~FlightPlan()
{

};

//
// flight sequence constraint functions
//

// adds body to flight plan
// body = name of body
// jd = julian date of flyby at body
// radius_min = minimum flyby radius (meters)
void FlightPlan::add_body(std::string body, Jdate jd, double radius_min, double d_scale, double orbits)
{
    bodies_.push_back(body);
    orbits_.push_back(orbits);
    t_.push_back(data_.ephemeris->get_ephemeris_time(jd));
    data_.radius_min.push_back(radius_min / data_.ephemeris->get_distance_scale());
    data_.d_scale.push_back(d_scale / data_.ephemeris->get_distance_scale());
}

void FlightPlan::add_start_time_constraint(Jdate start_time)
{
    data_.start_time = data_.ephemeris->get_ephemeris_time(start_time);
}

void FlightPlan::add_min_start_time_constraint(Jdate min_start_time)
{
    data_.min_start_time = data_.ephemeris->get_ephemeris_time(min_start_time);
}

void FlightPlan::add_min_flight_time_constraint(double min_time)
{
    data_.min_time = min_time * day_;
}

void FlightPlan::add_max_flight_time_constraint(double max_time)
{
    data_.max_time = max_time * day_;
}

void FlightPlan::add_max_c3_constraint(double max_c3)
{
    data_.max_c3 = max_c3 / pow(data_.ephemeris->get_velocity_scale(), 2);
}

void FlightPlan::add_arrival_eccentricity_constraint(double eccentricity)
{
    data_.eccentricity_arrival = eccentricity;
}

void FlightPlan::add_inclination_constraint(bool launch, double min, double max)
{
    if (launch)
    {
        data_.min_inclination_launch = astrodynamics::pi / 180.0 * min;
        data_.max_inclination_launch = astrodynamics::pi / 180.0 * max;
    }
    else
    {
        data_.min_inclination_arrival = astrodynamics::pi / 180.0 * min;
        data_.max_inclination_arrival = astrodynamics::pi / 180.0 * max;
    }    
}

// initializes patched conic model based on body list in flight plan
// eps = convergence criteria for kepler and lamert functions
void FlightPlan::init_conic_model(double eps)
{
    data_.num_bodies = bodies_.size();
    data_.num_phases = bodies_.size() - 1;
    data_.num_flybys = bodies_.size() - 2;

    // set sequence vector to body indicies
    for (auto& body : bodies_)
    {
        data_.sequence.push_back(data_.ephemeris->get_index_of_body(body));
    }

    // calculate launch and arrival velocity constants
    double orbital_radius_launch = data_.radius_min.front();
    double orbital_radius_arrival = data_.radius_min.back();

    double mu_body_launch = data_.ephemeris->get_mu(data_.sequence.front());
    double mu_body_arrival = data_.ephemeris->get_mu(data_.sequence.back());

    data_.vinf_squared_launch = 2.0 * mu_body_launch / orbital_radius_launch;
    data_.vcirc_launch = sqrt(mu_body_launch / orbital_radius_launch);

    data_.vinf_squared_arrival = 2.0 * mu_body_arrival / orbital_radius_arrival;
    data_.vell_arrival = sqrt((1.0 + data_.eccentricity_arrival) / (1.0 - data_.eccentricity_arrival) * mu_body_arrival / (orbital_radius_arrival / (1.0 - data_.eccentricity_arrival))); 

    // initialize time vector
    // first value set to specified time at first body
    // subsequent times added based on initlaization methods
    // for init_method = 'v' the outbound leg duration is specified
    // for init_method = 'd' the time is evenly split between legs
    // an additional dsm is placed halfway between each leg in time
    std::vector<double> ta;

    ta.push_back(t_.front());

    // add velocities to decision variable vector
    auto add_velocities = [](std::vector<double>& x, Eigen::Vector3d v)
    {
        x.push_back(v(0));
        x.push_back(v(1));
        x.push_back(v(2));
    };

    // calculate initial velocities
    for (int i = 0; i < data_.num_phases; i++)
    {
        double dt = t_[i + 1] - t_[i];

        // positions are placed at the center of the body at the specified time
        if (t_[i] < 0 || t_[i] > data_.ephemeris->get_max_time() || t_[i + 1] < 0 || t_[i + 1] > data_.ephemeris->get_max_time())
        {
            throw std::runtime_error("ephemeris time out of bounds.");
        }

        auto [ri, vbi] = data_.ephemeris->get_position_velocity(bodies_[i], t_[i]);
        auto [rf, vbf] = data_.ephemeris->get_position_velocity(bodies_[i + 1], t_[i + 1]);

        if (orbits_[i] > 0)
        {     
            double dt1 = dt * (orbits_[i] - 0.5) / orbits_[i];
            double dt2 = dt - dt1;
            ta.push_back(t_[i] + dt1 - 0.1 * dt2);
            ta.push_back(t_[i] + dt1);
            ta.push_back(t_[i] + dt1 + 0.1 * dt2);
            ta.push_back(t_[i + 1]);

            double a = pow(pow(dt / orbits_[i] / (2.0 * astrodynamics::pi), 2.0) * data_.ephemeris->get_muk(), 1.0 / 3.0);
            double v = sqrt(data_.ephemeris->get_muk() * (2.0 / ri.norm() - 1.0 / a));
            Eigen::Vector3d vi = v * vbi.normalized();
            
            auto [rm, vmm] = astrodynamics::kepler(ri, vi, dt1, data_.ephemeris->get_muk(), eps);
            auto [vmp, vf] = astrodynamics::lambert(rm, rf, dt2, data_.ephemeris->get_muk(), 1, home_body_normal_, eps);

            add_velocities(xa_, vi - vbi);
            add_velocities(xa_, vf - vbf); 
        }
        else
        {
            ta.push_back(t_[i] + 0.45 * dt);
            ta.push_back(t_[i] + 0.50 * dt);
            ta.push_back(t_[i] + 0.55 * dt);
            ta.push_back(t_[i + 1]);

            auto [vi, vf] = astrodynamics::lambert(ri, rf, dt, data_.ephemeris->get_muk(), 1, home_body_normal_, eps);

            add_velocities(xa_, vi - vbi);
            add_velocities(xa_, vf - vbf); 
        }     
    }

    // add time values to decision variable vector
    for (int i = 0; i < ta.size(); i++)
    {
        if (i == 0)
        {
            xa_.push_back(ta[i]);
        }
        else
        {
            xa_.push_back(ta[i] - ta[i - 1]);
        }  
    }

    // add dsm values to decision variable vector
    for (int i = 0; i < 9 * data_.num_phases; i++)
    {
        xa_.push_back(1e-4);
    }

    // add min flyby distance slack variables to decision variable vector
    for (int i = 0; i < data_.num_flybys; i++)
    {
        xa_.push_back(1e-4);
    }

    // add total flight time slack variable to decision variable vector
    double total_delta_time = t_.back() - t_.front();

    if (data_.max_time > total_delta_time)
    {
        xa_.push_back(data_.max_time - total_delta_time);
    }
    else
    {
        xa_.push_back(0.0);
    }  
}

// run the non-linear optimization for the patched conic model
void FlightPlan::run_conic_model(int max_eval, double eps, double eps_t, double eps_x)
{
    // calculate number of constraints and decision variables
    int m = 8 * data_.num_phases - 1;
    int n = 20 * data_.num_phases + 1;

    // initialize optimizer
    data_.eps = eps;

    std::vector<double> tol(m, eps_t);
    auto [lower_bounds, upper_bounds] = bounds_conic_model();

    double minf;
    opt_a_ = nlopt::opt("LD_SLSQP", n);

    opt_a_.set_lower_bounds(lower_bounds);
    opt_a_.set_upper_bounds(upper_bounds);
    opt_a_.set_min_objective(objective_conic_model, &data_);
    opt_a_.add_equality_mconstraint(constraints_conic_model, &data_, tol);
    opt_a_.set_maxeval(max_eval);
    opt_a_.set_xtol_abs(eps_x);
    opt_a_.optimize(xa_, minf);
}

// set conic solution
void FlightPlan::set_conic_solution(std::vector<double> x)
{
    xa_ = x;
}

// creates output for conic model
FlightPlan::Result FlightPlan::output_conic_result(double eps)
{
    Result result;

    int m = 8 * data_.num_phases - 1;
    int n = 20 * data_.num_phases + 1;

    data_.eps = eps;

    result.nlopt_constraints.resize(m);
    result.r.resize(3);
    result.v.resize(3);
    result.rb.resize(data_.num_bodies, std::vector<std::vector<double>>(3));
    result.vb.resize(data_.num_bodies, std::vector<std::vector<double>>(3));

    try
    {
        result.nlopt_code = opt_a_.last_optimize_result();
        result.nlopt_value = opt_a_.last_optimum_value();
        result.nlopt_num_evals = opt_a_.get_numevals();
    }
    catch (const std::exception& e)
    {

    }

    result.nlopt_solution = xa_;

    constraints_conic_model(m, result.nlopt_constraints.data(), n, xa_.data(), NULL, &data_);

    result.time_scale = data_.ephemeris->get_time_scale();
    result.distance_scale = data_.ephemeris->get_distance_scale();
    result.velocity_scale = data_.ephemeris->get_velocity_scale();

    result.sequence = data_.sequence;
    result.bodies = data_.ephemeris->get_bodies();
    result.mu = data_.ephemeris->get_mu();

    // initialize data
    int index_v0 = 0;
    int index_t0 = 6 * data_.num_phases;
    int index_dt = 6 * data_.num_phases + 1;
    int index_dv = 10 * data_.num_phases + 1;
    int index_sl = 19 * data_.num_phases + 1;

    std::vector<double> t(4 * data_.num_phases + 1);
    std::vector<Eigen::Vector3d> rb(data_.num_phases + 1);
    std::vector<Eigen::Vector3d> vb(data_.num_phases + 1);
    std::vector<Eigen::Vector3d> v0(2 * data_.num_phases);
    std::vector<Eigen::Vector3d> r1(4 * data_.num_phases);
    std::vector<Eigen::Vector3d> v1(4 * data_.num_phases);
    std::vector<Eigen::Vector3d> dv(3 * data_.num_phases);

    t[0] = xa_[index_t0];
    for (int i = 0; i < 4 * data_.num_phases; i++)
    {
        t[i + 1] = t[i] + xa_[index_dt + i];
    }

    for (int i = 0; i < data_.num_phases; i++)
    {
        int bi = data_.sequence[i];
        int bf = data_.sequence[i + 1];
        int iv0 = index_v0 + 6 * i;
        int idv = index_dv + 9 * i;
        double ti = t[4 * i];
        double tf = t[4 * (i + 1)];

        if (i == 0)
        {
            std::tie(rb[0], vb[0]) = data_.ephemeris->get_position_velocity(bi, ti);
        }

        std::tie(rb[i + 1], vb[i + 1]) = data_.ephemeris->get_position_velocity(bf, tf);

        v0[2 * i] << xa_[iv0], xa_[iv0 + 1], xa_[iv0 + 2];
        v0[2 * i + 1] << xa_[iv0 + 3], xa_[iv0 + 4], xa_[iv0 + 5];

        v0[2 * i] += vb[i];
        v0[2 * i + 1] += vb[i + 1];

        dv[3 * i] << xa_[idv], xa_[idv + 1], xa_[idv + 2];
        dv[3 * i + 1] << xa_[idv + 3], xa_[idv + 4], xa_[idv + 5];
        dv[3 * i + 2] << xa_[idv + 6], xa_[idv + 7], xa_[idv + 8];
    }

    // propagate trajectories
    for (int i = 0; i < data_.num_phases; i++)
    {
        std::tie(r1[4 * i], v1[4 * i]) = astrodynamics::kepler(rb[i], v0[2 * i], xa_[index_dt + 4 * i], data_.ephemeris->get_muk(), eps);
        v1[4 * i] += dv[3 * i];
        std::tie(r1[4 * i + 1], v1[4 * i + 1]) = astrodynamics::kepler(r1[4 * i], v1[4 * i], xa_[index_dt + 4 * i + 1], data_.ephemeris->get_muk(), eps);
        v1[4 * i + 1] += dv[3 * i + 1];

        std::tie(r1[4 * i + 3], v1[4 * i + 3]) = astrodynamics::kepler(rb[i + 1], v0[2 * i + 1], -xa_[index_dt + 4 * i + 3], data_.ephemeris->get_muk(), eps);
        v1[4 * i + 3] -= dv[3 * i + 2];
        std::tie(r1[4 * i + 2], v1[4 * i + 2]) = astrodynamics::kepler(r1[4 * i + 3], v1[4 * i + 3], -xa_[index_dt + 4 * i + 2], data_.ephemeris->get_muk(), eps);
    }

    // output function, outputs 100 data points per year
    auto output_kepler = [this, &result](int phase, int leg, Eigen::Vector3d r0, Eigen::Vector3d v0, double ti, double tf)
    {
        int steps = 50;

        for (int i = 0; i <= steps; i++)
        {
            double dt = (tf - ti) * static_cast<double>(i) / steps;
            auto [r1, v1] = astrodynamics::kepler(r0, v0, dt, data_.ephemeris->get_muk(), 1e-8);
            double t = ti + dt;
            Jdate jd = data_.ephemeris->get_jdate(t);
            result.julian_time.push_back(jd.get_julian_date());
            result.kerbal_time.push_back(jd.get_kerbal_time());
            result.phase.push_back(phase);
            result.leg.push_back(leg);
            if (leg < 2)
            {
                result.body.push_back(data_.sequence[phase]);
            }
            else
            {
                result.body.push_back(data_.sequence[phase + 1]);
            }

            result.r[0].push_back(r1(0));
            result.r[1].push_back(r1(1));
            result.r[2].push_back(r1(2));
            result.v[0].push_back(v1(0));
            result.v[1].push_back(v1(1));
            result.v[2].push_back(v1(2));

            for (int j = 0; j < data_.num_bodies; j++)
            {
                int b = data_.sequence[j];
                auto [rb, vb] = data_.ephemeris->get_position_velocity(b, t);
                result.rb[j][0].push_back(rb(0));
                result.rb[j][1].push_back(rb(1));
                result.rb[j][2].push_back(rb(2));
                result.vb[j][0].push_back(vb(0));
                result.vb[j][1].push_back(vb(1));
                result.vb[j][2].push_back(vb(2));
            }
        }
    };

    // output loop
    for (int i = 0; i < data_.num_phases; i++)
    {
        output_kepler(i, 0, rb[i], v0[2 * i], t[4 * i], t[4 * i + 1]);
        output_kepler(i, 1, r1[4 * i], v1[4 * i], t[4 * i + 1], t[4 * i + 2]);
        output_kepler(i, 2, r1[4 * i + 2], v1[4 * i + 2], t[4 * i + 2], t[4 * i + 3]);
        output_kepler(i, 3, r1[4 * i + 3], v1[4 * i + 3], t[4 * i + 3], t[4 * i + 4]);
    }

    return result;
}

// set bounds for decisions variables for patched conic model
std::tuple<std::vector<double>, std::vector<double>> FlightPlan::bounds_conic_model()
{
    std::vector<double> lower_bounds;
    std::vector<double> upper_bounds;

    // velocity bounds
    for (int i = 0; i < bodies_.size() - 1; i++)
    {
        auto [rbi, vbi] = data_.ephemeris->get_position_velocity(bodies_[i], 0.0);
        auto [rbf, vbf] = data_.ephemeris->get_position_velocity(bodies_[i + 1], 0.0);

        double ai = astrodynamics::orbit_semi_major_axis(rbi, vbi, data_.ephemeris->get_muk());
        double af = astrodynamics::orbit_semi_major_axis(rbf, vbf, data_.ephemeris->get_muk());

        double vesci = sqrt(2.0 * data_.ephemeris->get_muk() / ai);
        double vescf = sqrt(2.0 * data_.ephemeris->get_muk() / af);

        for (int j = 0; j < 3; j++)
        {
            lower_bounds.push_back(-vesci);
            upper_bounds.push_back(vesci);
        }

        for (int j = 0; j < 3; j++)
        {
            lower_bounds.push_back(-vescf);
            upper_bounds.push_back(vescf);
        }
    }

    // start time must be greater than zero and within a year of initial guess
    if (data_.min_start_time > 0)
    {
        lower_bounds.push_back(data_.min_start_time);
    }
    else if (t_.front() > year_)
    {
        lower_bounds.push_back(t_.front() - year_);
    }
    else
    {
        lower_bounds.push_back(0.0);
    }
    
    upper_bounds.push_back(t_.front() + year_);

    // time step bounds
    for (int i = 0; i < data_.num_phases; i++)
    {
        double muk = data_.ephemeris->get_muk();
        double mui = data_.ephemeris->get_mu(bodies_[i]);
        auto [rbi, vbi] = data_.ephemeris->get_position_velocity(bodies_[i], 0.0);
        double ai = astrodynamics::orbit_semi_major_axis(rbi, vbi, muk);

        double ri = ai * pow(2.0, -0.2) * pow(mui / muk, 0.4);
        double ti = sqrt(pow(1.5 * ri, 3) / mui / 4.5);

        double muf = data_.ephemeris->get_mu(bodies_[i + 1]);
        auto [rbf, vbf] = data_.ephemeris->get_position_velocity(bodies_[i + 1], 0.0);
        double af = astrodynamics::orbit_semi_major_axis(rbf, vbf, muk);

        double rf = af * pow(2.0, -0.2) * pow(muf / muk, 0.4);
        double tf = sqrt(pow(rf, 3) / muf / 4.5);

        lower_bounds.push_back(ti);
        lower_bounds.push_back(0.01 * day_);
        lower_bounds.push_back(0.01 * day_);
        lower_bounds.push_back(tf);
        upper_bounds.push_back(HUGE_VAL);
        upper_bounds.push_back(HUGE_VAL);
        upper_bounds.push_back(HUGE_VAL);
        upper_bounds.push_back(HUGE_VAL);
    }

    // dsm bounds
    for (int i = 0; i < 9 * data_.num_phases; i++)
    {
        lower_bounds.push_back(-1500.0 / data_.ephemeris->get_velocity_scale());
        upper_bounds.push_back(1500.0 / data_.ephemeris->get_velocity_scale());
    }

    // slack variable bounds
    for (int i = 0; i < data_.num_flybys + 1; i++)
    {
        lower_bounds.push_back(0.0);
        upper_bounds.push_back(HUGE_VAL);
    }

    return { lower_bounds, upper_bounds };
}

// calculate constraints and derivatives of conic model for nlopt
void FlightPlan::constraints_conic_model(unsigned m, double *result, unsigned n, const double* x, double* grad, void* f_data)
{
    try
    {
        FlightPlanData* data = reinterpret_cast<FlightPlanData*>(f_data);

        // initialize data
        int index_v0 = 0;
        int index_t0 = 6 * data->num_phases;
        int index_dt = 6 * data->num_phases + 1;
        int index_dv = 10 * data->num_phases + 1;
        int index_sl = 19 * data->num_phases + 1;

        std::vector<double> t(4 * data->num_phases + 1);
        std::vector<Eigen::Vector3d> rb(data->num_bodies);
        std::vector<Eigen::Vector3d> vb(data->num_bodies);
        std::vector<Eigen::Vector3d> ab(data->num_bodies);
        std::vector<Eigen::Vector3d> v0(2 * data->num_phases);
        std::vector<Eigen::Vector3d> r1(4 * data->num_phases);
        std::vector<Eigen::Vector3d> v1(4 * data->num_phases);
        std::vector<Eigen::Matrix<double, 6, 1>> a1(4 * data->num_phases);
        std::vector<Eigen::Vector3d> dv(3 * data->num_phases);
        std::vector<Eigen::Matrix<double, 6, 6>> stm(4 * data->num_phases);

        t[0] = x[index_t0];
        for (int i = 0; i < 4 * data->num_phases; i++)
        {
            t[i + 1] = t[i] + x[index_dt + i];
        }

        for (int i = 0; i < data->num_phases; i++)
        {
            int bi = data->sequence[i];
            int bf = data->sequence[i + 1];
            int iv0 = index_v0 + 6 * i;
            int idv = index_dv + 9 * i;
            double ti = t[4 * i];
            double tf = t[4 * (i + 1)];

            if (grad)
            {
                if (i == 0)
                {
                    std::tie(rb[0], vb[0], ab[0]) = data->ephemeris->get_position_velocity_acceleration(bi, ti);
                }

                std::tie(rb[i + 1], vb[i + 1], ab[i + 1]) = data->ephemeris->get_position_velocity_acceleration(bf, tf);
            }
            else
            {
                if (i == 0)
                {
                    std::tie(rb[0], vb[0]) = data->ephemeris->get_position_velocity(bi, ti);
                }

                std::tie(rb[i + 1], vb[i + 1]) = data->ephemeris->get_position_velocity(bf, tf);
            }
            
            v0[2 * i] << x[iv0], x[iv0 + 1], x[iv0 + 2];
            v0[2 * i + 1] << x[iv0 + 3], x[iv0 + 4], x[iv0 + 5];

            v0[2 * i] += vb[i];
            v0[2 * i + 1] += vb[i + 1];

            dv[3 * i] << x[idv], x[idv + 1], x[idv + 2];
            dv[3 * i + 1] << x[idv + 3], x[idv + 4], x[idv + 5];
            dv[3 * i + 2] << x[idv + 6], x[idv + 7], x[idv + 8];
        }

        // propagate trajectories
        if (grad)
        {
            for (int i = 0; i < data->num_phases; i++)
            {
                std::tie(r1[4 * i], v1[4 * i], a1[4 * i], stm[4 * i]) = astrodynamics::kepler_stm(rb[i], v0[2 * i], x[index_dt + 4 * i], data->ephemeris->get_muk(), data->eps);
                v1[4 * i] += dv[3 * i];
                std::tie(r1[4 * i + 1], v1[4 * i + 1], a1[4 * i + 1], stm[4 * i + 1]) = astrodynamics::kepler_stm(r1[4 * i], v1[4 * i], x[index_dt + 4 * i + 1], data->ephemeris->get_muk(), data->eps);
                v1[4 * i + 1] += dv[3 * i + 1];

                std::tie(r1[4 * i + 3], v1[4 * i + 3], a1[4 * i + 3], stm[4 * i + 3]) = astrodynamics::kepler_stm(rb[i + 1], v0[2 * i + 1], -x[index_dt + 4 * i + 3], data->ephemeris->get_muk(), data->eps);
                v1[4 * i + 3] -= dv[3 * i + 2];
                std::tie(r1[4 * i + 2], v1[4 * i + 2], a1[4 * i + 2], stm[4 * i + 2]) = astrodynamics::kepler_stm(r1[4 * i + 3], v1[4 * i + 3], -x[index_dt + 4 * i + 2], data->ephemeris->get_muk(), data->eps);
            }
        }
        else
        {
            for (int i = 0; i < data->num_phases; i++)
            {
                std::tie(r1[4 * i], v1[4 * i]) = astrodynamics::kepler(rb[i], v0[2 * i], x[index_dt + 4 * i], data->ephemeris->get_muk(), data->eps);
                v1[4 * i] += dv[3 * i];
                std::tie(r1[4 * i + 1], v1[4 * i + 1]) = astrodynamics::kepler(r1[4 * i], v1[4 * i], x[index_dt + 4 * i + 1], data->ephemeris->get_muk(), data->eps);
                v1[4 * i + 1] += dv[3 * i + 1];

                std::tie(r1[4 * i + 3], v1[4 * i + 3]) = astrodynamics::kepler(rb[i + 1], v0[2 * i + 1], -x[index_dt + 4 * i + 3], data->ephemeris->get_muk(), data->eps);
                v1[4 * i + 3] -= dv[3 * i + 2];
                std::tie(r1[4 * i + 2], v1[4 * i + 2]) = astrodynamics::kepler(r1[4 * i + 3], v1[4 * i + 3], -x[index_dt + 4 * i + 2], data->ephemeris->get_muk(), data->eps);
            }
        }

        // populate result vector
        double* result_ptr = result;

        // match point constraints
        for (int i = 0; i < data->num_phases; i++)
        {
            Eigen::Vector3d drmp = r1[4 * i + 2] - r1[4 * i + 1];
            Eigen::Vector3d dvmp = v1[4 * i + 2] - v1[4 * i + 1];

            *result_ptr++ = drmp(0);
            *result_ptr++ = drmp(1);
            *result_ptr++ = drmp(2);
            *result_ptr++ = dvmp(0);
            *result_ptr++ = dvmp(1);
            *result_ptr++ = dvmp(2);
        }

        // minimum flyby distance and vinf constraints
        for (int i = 0; i < data->num_flybys; i++)
        {
            int bi = i + 1;
            Eigen::Vector3d vif = v0[2 * bi] - vb[bi];
            Eigen::Vector3d vib = v0[2 * bi - 1] - vb[bi];
            double sif2 = vif.squaredNorm();
            double sif = sqrt(sif2);
            double sib = vib.norm();
            double delta = acos(vif.dot(vib) / sif / sib);
            double alt = data->ephemeris->get_mu(data->sequence[bi]) / sif2 * (1.0 / sin(delta / 2.0) - 1.0);
            double dvi = sif - sib;
            *result_ptr++ = alt - data->radius_min[bi] - x[index_sl + i];
            *result_ptr++ = dvi;
        }

        // maximum time of flight constraint
        *result_ptr++ = t.back() - t.front() - data->max_time + x[index_sl + data->num_flybys];

        // calculate gradient
        if (grad)
        {
            const double* x_ptr = x;
            Eigen::MatrixXd mgrad = Eigen::MatrixXd::Zero(m, n);

            for (int i = 0; i < data->num_phases; i++)
            {
                mgrad.block<6, 3>(6 * i, 6 * i) = (-stm[4 * i + 1] * stm[4 * i]).block<6, 3>(0, 3);
                mgrad.block<6, 3>(6 * i, 6 * i + 3) = (stm[4 * i + 2] * stm[4 * i + 3]).block<6, 3>(0, 3);

                Eigen::Matrix<double, 6, 1> xbp0, xbp1;
                xbp0 << vb[i](0),  vb[i](1), vb[i](2), ab[i](0),  ab[i](1), ab[i](2);
                xbp1 << vb[i + 1](0),  vb[i + 1](1), vb[i + 1](2), ab[i + 1](0),  ab[i + 1](1), ab[i + 1](2);

                Eigen::Matrix<double, 6, 1> dmp_dst0 = -stm[4 * i + 1] * stm[4 * i] * xbp0;
                Eigen::Matrix<double, 6, 1> dmp_dst1 = stm[4 * i + 2] * stm[4 * i + 3] * xbp1;

                for (int j = 0; j < 4 * i + 1; j++)
                {
                    mgrad.block<6, 1>(6 * i, 6 * data->num_phases + j) = dmp_dst0 + dmp_dst1;
                }
                
                mgrad.block<6, 1>(6 * i, 6 * data->num_phases + 4 * i + 1) = -stm[4 * i + 1] * a1[4 * i] + dmp_dst1;
                mgrad.block<6, 1>(6 * i, 6 * data->num_phases + 4 * i + 2) = -a1[4 * i + 1] + dmp_dst1;
                mgrad.block<6, 1>(6 * i, 6 * data->num_phases + 4 * i + 3) = -a1[4 * i + 2] + dmp_dst1;
                mgrad.block<6, 1>(6 * i, 6 * data->num_phases + 4 * i + 4) = stm[4 * i + 2] * (-a1[4 * i + 3] + stm[4 * i + 3] * xbp1);

                mgrad.block<6, 3>(6 * i, 10 * data->num_phases + i * 9 + 1) = -stm[4 * i + 1].block<6, 3>(0, 3);
                mgrad.block<3, 3>(6 * i + 3, 10 * data->num_phases + i * 9 + 4) = -Eigen::Matrix3d::Identity();
                mgrad.block<6, 3>(6 * i, 10 * data->num_phases + i * 9 + 7) = -stm[4 * i + 2].block<6, 3>(0, 3);
            }

            for (int i = 0; i < data->num_flybys; i++)
            {
                int bi = i + 1;
                Eigen::Vector3d vif = v0[2 * bi] - vb[bi];
                Eigen::Vector3d vib = v0[2 * bi - 1] - vb[bi];
                double sif2 = vif.squaredNorm();
                double sif = sqrt(sif2);
                double sib = vib.norm();
                double delta = acos(vif.dot(vib) / sif / sib);

                double alpha = cos(delta / 2.0) / (cos(delta) - 1.0) / sif2;
                double beta = 1.0 / sin(delta / 2.0) - 1.0;
                double psi = alpha / sif / sib / sqrt(1.0 - pow(vif.dot(vib) / sif / sib, 2.0));
                double xip = 2.0 / (sif2 * sif2) * beta - psi * vif.dot(vib) / sif2;
                double xim = -psi * vif.dot(vib) / (sib * sib);

                mgrad(6 * data->num_phases + 2 * i, 6 * bi - 3) = -data->ephemeris->get_mu(data->sequence[bi]) * (xim * vib[0] + psi * vif[0]);
                mgrad(6 * data->num_phases + 2 * i, 6 * bi - 2) = -data->ephemeris->get_mu(data->sequence[bi]) * (xim * vib[1] + psi * vif[1]);
                mgrad(6 * data->num_phases + 2 * i, 6 * bi - 1) = -data->ephemeris->get_mu(data->sequence[bi]) * (xim * vib[2] + psi * vif[2]);
                mgrad(6 * data->num_phases + 2 * i, 6 * bi - 0) = -data->ephemeris->get_mu(data->sequence[bi]) * (xip * vif[0] + psi * vib[0]);
                mgrad(6 * data->num_phases + 2 * i, 6 * bi + 1) = -data->ephemeris->get_mu(data->sequence[bi]) * (xip * vif[1] + psi * vib[1]);
                mgrad(6 * data->num_phases + 2 * i, 6 * bi + 2) = -data->ephemeris->get_mu(data->sequence[bi]) * (xip * vif[2] + psi * vib[2]);

                mgrad(6 * data->num_phases + 2 * i + 1, 6 * bi - 3) = -vib(0) / sib;
                mgrad(6 * data->num_phases + 2 * i + 1, 6 * bi - 2) = -vib(1) / sib;
                mgrad(6 * data->num_phases + 2 * i + 1, 6 * bi - 1) = -vib(2) / sib;
                mgrad(6 * data->num_phases + 2 * i + 1, 6 * bi - 0) = vif(0) / sif;
                mgrad(6 * data->num_phases + 2 * i + 1, 6 * bi + 1) = vif(1) / sif;
                mgrad(6 * data->num_phases + 2 * i + 1, 6 * bi + 2) = vif(2) / sif;

                mgrad(6 * data->num_phases + 2 * i, n - data->num_phases + i) = -1;
            }

            for (int i = 0; i < 4 * data->num_phases; i++)
            {
                mgrad(m - 1, 6 * data->num_phases + 1 + i) = 1;
            }
            
            mgrad(m - 1, n - 1) = 1;

            for (int j = 0; j < m; j++)
            {
                for (int i = 0; i < n; i++)
                {
                    grad[j * n + i] = mgrad(j, i);
                }
            }
        }
    }
    catch(const std::exception& e)
    {
        std::cerr << "error in conic constraint function:" << '\n';
        std::cerr << e.what() << '\n';
        throw e;
    }
}

// calculate objective function of patched conic model for nlopt
double FlightPlan::objective_conic_model(unsigned n, const double* x, double* grad, void* f_data)
{
    try
    {
        FlightPlanData* data = reinterpret_cast<FlightPlanData*>(f_data);

        // initialize data
        int index_t0 = 6 * data->num_phases;
        int index_dt = 6 * data->num_phases + 1;
        int index_dv = 10 * data->num_phases + 1;

        std::vector<Eigen::Vector3d> dv(3 * data->num_phases);

        for (int i = 0; i < data->num_phases; i++)
        {
            int idv = index_dv + 9 * i;
            dv[3 * i + 0] << x[idv + 0], x[idv + 1], x[idv + 2];
            dv[3 * i + 1] << x[idv + 3], x[idv + 4], x[idv + 5];
            dv[3 * i + 2] << x[idv + 6], x[idv + 7], x[idv + 8];
        }

        // calculate launch and arrival delta velocity
        Eigen::Vector3d vli;
        Eigen::Vector3d vai;

        vli << x[0], x[1], x[2];
        vai << x[6 * (data->num_phases - 1) + 3], x[6 * (data->num_phases - 1) + 4], x[6 * (data->num_phases - 1) + 5];

        double dvl = sqrt(vli.squaredNorm() + data->vinf_squared_launch) - data->vcirc_launch;
        double dva = sqrt(vai.squaredNorm() + data->vinf_squared_arrival) - data->vell_arrival;

        double dvt = dvl;

        if (data->eccentricity_arrival >= 0)
        {
            dvt += dva;
        }

        // add dsm delta velocity
        for (int i = 0; i < dv.size(); i++)
        {
            dvt += dv[i].norm();
        }

        // calculate gradient numerically 
        if (grad)
        {
            grad[0] = vli[0] / sqrt(vli.squaredNorm() + data->vinf_squared_launch);
            grad[1] = vli[1] / sqrt(vli.squaredNorm() + data->vinf_squared_launch);
            grad[2] = vli[2] / sqrt(vli.squaredNorm() + data->vinf_squared_launch);

            if (data->eccentricity_arrival >= 0)
            {
                grad[index_t0 - 3] = vai[0] / sqrt(vai.squaredNorm() + data->vinf_squared_arrival);
                grad[index_t0 - 2] = vai[1] / sqrt(vai.squaredNorm() + data->vinf_squared_arrival);
                grad[index_t0 - 1] = vai[2] / sqrt(vai.squaredNorm() + data->vinf_squared_arrival);
            }
            else
            {
                grad[index_t0 - 3] = 0.0;
                grad[index_t0 - 2] = 0.0;
                grad[index_t0 - 1] = 0.0;
            }

            for (int i = 0; i < data->num_phases; i++)
            {
                int idv = index_dv + 9 * i;

                grad[idv + 0] = x[idv + 0] / dv[3 * i + 0].norm();
                grad[idv + 1] = x[idv + 1] / dv[3 * i + 0].norm();
                grad[idv + 2] = x[idv + 2] / dv[3 * i + 0].norm();
                grad[idv + 3] = x[idv + 3] / dv[3 * i + 1].norm();
                grad[idv + 4] = x[idv + 4] / dv[3 * i + 1].norm();
                grad[idv + 5] = x[idv + 5] / dv[3 * i + 1].norm();
                grad[idv + 6] = x[idv + 6] / dv[3 * i + 2].norm();
                grad[idv + 7] = x[idv + 7] / dv[3 * i + 2].norm();
                grad[idv + 8] = x[idv + 8] / dv[3 * i + 2].norm();
            }

            for (int i = 3; i < index_t0 - 3; i++)
            {
                grad[i] = 0.0;
            }

            for (int i = index_t0; i < index_dv; i++)
            {
                grad[i] = 0.0;
            }

            for (int i = index_dv + 9 * data->num_phases; i < n; i++)
            {
                grad[i] = 0.0;
            }
        }

        return dvt;
    }
    catch (const std::exception& e)
    {
        std::cerr << "error in conic objective function:" << '\n';
        std::cerr << e.what() << '\n';
        throw e;
    }
}

//
// nbody functions
//

// initializes numerical model based on output from converged patched conic solution
// eps = convergence criteria
void FlightPlan::init_nbody_model(double eps)
{
    data_.eps = eps;

    // calculate time and velocity scale factors for each body
    for (int i = 0; i < data_.num_bodies; i++)
    {
        data_.t_scale.push_back(sqrt(pow(data_.d_scale[i], 3.0) / data_.ephemeris->get_mu(data_.sequence[i])));
        data_.v_scale.push_back(data_.d_scale[i] / data_.t_scale[i]);
    }

    // calculate number of moon flybys
    data_.num_moon_flybys = 0;
    for (int i = 0; i < data_.num_bodies; i++)
    {
        if (bodies_[i] == home_body_)
        {
            if (i == 0 || i == data_.num_bodies - 1)
            {
                data_.num_moon_flybys++;
            }
            else
            {
                data_.num_moon_flybys += 2;
            }
            
        }
    }

    // calculate sphere of influence sizes
    for (int i = 0; i < data_.num_bodies; i++)
    {
        double muk = data_.ephemeris->get_muk();
        double mui = data_.ephemeris->get_mu(bodies_[i]);
        auto [rb, vb] = data_.ephemeris->get_position_velocity(bodies_[i], 0.0);
        double a = astrodynamics::orbit_semi_major_axis(rb, vb, muk);

        data_.radius_soi.push_back(a * pow(2.0, -0.2) * pow(mui / muk, 0.4));
    }

    // parse patched conic solution
    int index_v0 = 0;
    int index_t0 = 6 * data_.num_phases;
    int index_dt = 6 * data_.num_phases + 1;
    int index_dv = 10 * data_.num_phases + 1;
    int index_sl = 19 * data_.num_phases + 1;

    std::vector<double> ta(4 * data_.num_phases + 1);
    std::vector<Eigen::Vector3d> vb(data_.num_phases + 1);
    std::vector<Eigen::Vector3d> v0(2 * data_.num_phases);
    std::vector<Eigen::Vector3d> dv(3 * data_.num_phases);

    std::vector<Eigen::Vector3d> rp(data_.num_bodies);
    std::vector<Eigen::Vector3d> vp(data_.num_bodies);

    ta[0] = xa_[index_t0];
    for (int i = 0; i < 4 * data_.num_phases; i++)
    {
        ta[i + 1] = ta[i] + xa_[index_dt + i];
    }

    for (int i = 0; i < data_.num_phases; i++)
    {
        int bi = data_.sequence[i];
        int bf = data_.sequence[i + 1];
        int iv0 = index_v0 + 6 * i;
        int idv = index_dv + 9 * i;
        double ti = ta[4 * i];
        double tf = ta[4 * (i + 1)];

        if (i == 0)
        {
            vb[0] = data_.ephemeris->get_velocity(bi, ti);
        }

        vb[i + 1] = data_.ephemeris->get_velocity(bf, tf);

        v0[2 * i] << xa_[iv0], xa_[iv0 + 1], xa_[iv0 + 2];
        v0[2 * i + 1] << xa_[iv0 + 3], xa_[iv0 + 4], xa_[iv0 + 5];

        v0[2 * i] += vb[i];
        v0[2 * i + 1] += vb[i + 1];

        dv[3 * i] << xa_[idv], xa_[idv + 1], xa_[idv + 2];
        dv[3 * i + 1] << xa_[idv + 3], xa_[idv + 4], xa_[idv + 5];
        dv[3 * i + 2] << xa_[idv + 6], xa_[idv + 7], xa_[idv + 8];
    }

    // calculate periapsis states
    std::tie(rp[0], vp[0]) = astrodynamics::planet_orbit_periapsis(data_.ephemeris, v0[0], data_.radius_min[0], 1, data_.sequence[0], ta[0]);
        
    for (int i = 1; i < data_.num_bodies - 1; i++)
    {
        std::tie(rp[i], vp[i]) = astrodynamics::planet_flyby_periapsis(data_.ephemeris, v0[2 * i - 1], v0[2 * i], data_.sequence[i], ta[4 * i]);
    }
    
    std::tie(rp[data_.num_bodies - 1], vp[data_.num_bodies - 1]) = astrodynamics::planet_orbit_periapsis(data_.ephemeris, v0[2 * data_.num_phases - 1], data_.radius_min[data_.num_bodies - 1], -1, data_.sequence[data_.num_bodies - 1], ta[4 * (data_.num_bodies - 1)]);
    
    // calculate soi delta time
    std::vector<double> dtsoi;

    for (int i = 0; i < data_.num_bodies; i++)
    {
        dtsoi.push_back(astrodynamics::hyperbolic_orbit_time_at_distance(rp[i], vp[i], data_.radius_soi[i], data_.ephemeris->get_mu(bodies_[i])));
    }

    // converts state to spherical coordinates and adds to x vector
    auto push_back_state = [](std::vector<double>& x, Eigen::Vector3d r, Eigen::Vector3d v, bool radius)
    {         
        if (radius)
        {
            x.push_back(r.norm());
        }

        x.push_back(atan2(r(2), r(0)));
        x.push_back(asin(r(1) / r.norm()));
        x.push_back(v.norm());
        x.push_back(atan2(v(2), v(0)));
        x.push_back(asin(v(1) / v.norm()));
    };

    // delta time vector
    std::vector<double> dtn;

    for (int i = 0; i < data_.num_phases; i++)
    {
        // calculate state at soi
        auto [rsoif, vsoif] = astrodynamics::kepler_s(rp[i], vp[i], dtsoi[i], data_.ephemeris->get_mu(bodies_[i]), eps);
        auto [rsoib, vsoib] = astrodynamics::kepler_s(rp[i + 1], vp[i + 1], -dtsoi[i + 1], data_.ephemeris->get_mu(bodies_[i + 1]), eps);

        // add start state
        if (i == 0)
        {
            push_back_state(xn_, rp[i], vp[i], false);
        }

        // if start body is home body add state at moon orbit radius
        if (bodies_[i] == home_body_)
        {
            double dtm = astrodynamics::hyperbolic_orbit_time_at_distance(rp[i], vp[i], data_.moon_mean_distance, data_.ephemeris->get_mu(bodies_[i]));
            auto [rmf, vmf] = astrodynamics::kepler_s(rp[i], vp[i], dtm, data_.ephemeris->get_mu(bodies_[i]), eps);

            push_back_state(xn_, rmf, vmf, false);
            dtn.push_back(dtm);
            dtn.push_back(dtsoi[i] - dtm);
        }
        else
        {
            dtn.push_back(dtsoi[i]);
        }

        // add state at start body soi
        push_back_state(xn_, rsoif, vsoif, false);
        
        // add dsm times, subtract soi time from first and last
        for (int j = 0; j < 4; j++)
        {
            double dtdsm = xa_[index_dt + 4 * i + j];

            if (j == 0)
            {
                dtdsm -= dtsoi[i];
            }

            if (j == 3)
            {
                dtdsm -= dtsoi[i + 1];
            }

            dtn.push_back(dtdsm);
        }

        // add state at end body soi
        push_back_state(xn_, rsoib, vsoib, false);

        // if end body is home body add state at moon orbit radius
        if (bodies_[i + 1] == home_body_)
        {
            double dtm = astrodynamics::hyperbolic_orbit_time_at_distance(rp[i + 1], vp[i + 1], data_.moon_mean_distance, data_.ephemeris->get_mu(bodies_[i + 1]));
            auto [rmb, vmb] = astrodynamics::kepler_s(rp[i + 1], vp[i + 1], -dtm, data_.ephemeris->get_mu(bodies_[i + 1]), eps);

            push_back_state(xn_, rmb, vmb, false);
            dtn.push_back(dtsoi[i + 1] - dtm);
            dtn.push_back(dtm);
        }
        else
        {
            dtn.push_back(dtsoi[i + 1]);
        }

        // add state at end body periapsis
        if (i == data_.num_phases - 1)
        {
            push_back_state(xn_, rp[i + 1], vp[i + 1], false);
        }
        else
        {
            push_back_state(xn_, rp[i + 1], vp[i + 1], true);
        }
    }

    // add variables to decision variable vector
    xn_.push_back(ta[0]);

    for (int i = 0; i < dtn.size(); i++)
    {
        xn_.push_back(dtn[i]);
    }

    // add dsm unit vectors to decision variable vector
    for (int i = 0; i < data_.num_phases; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            xn_.push_back(dv[3 * i + j].normalized()(0));
            xn_.push_back(dv[3 * i + j].normalized()(1));
            xn_.push_back(dv[3 * i + j].normalized()(2));
        }
    }

    // add dsm magnitudes to decision variable vector
    for (int i = 0; i < data_.num_phases; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            xn_.push_back(dv[3 * i + j].norm());
        }
    }

    // add slack variables to decision variable vector

    // lunar distance slack variables
    for (int i = 0; i < data_.num_moon_flybys; i++)
    {
        xn_.push_back(0.0);
    }

    // inclination slack variables
    xn_.push_back(0.0);
    xn_.push_back(0.0);
    xn_.push_back(0.0);
    xn_.push_back(0.0);

    // max time slack variable
    if (data_.max_time - (ta[4 * data_.num_phases] - ta[0]) > 0)
    {
        xn_.push_back(data_.max_time - (ta[4 * data_.num_phases] - ta[0]));
    }
    else
    {
        xn_.push_back(0.0);
    }
    

    // min time slack variable
    if (data_.min_time > 0)
    {
        if ((ta[4 * data_.num_phases] - ta[0]) - data_.min_time > 0)
        {
            xn_.push_back((ta[4 * data_.num_phases] - ta[0]) - data_.min_time);
        }
        else
        {
            xn_.push_back(0.0);
        }  
    }

    // c3 slack variable
    if (data_.max_c3 > 0)
    {
        xn_.push_back(0.0);
    }  
}

// set nbody solution
void FlightPlan::set_nbody_solution(std::vector<double> x)
{
    xn_ = x;
}

// run the non-linear optimization for the nbody model
void FlightPlan::run_nbody_model(int max_eval, double eps, double eps_t, double eps_f)
{
    // calculate number of constraints and decision variables
    int n = 34 * (data_.num_bodies - 1) + 7 * data_.num_moon_flybys + 10;
    int m = 22 * (data_.num_bodies - 1) + 7 * data_.num_moon_flybys + 6;

    if (data_.start_time > 0)
    {
        m++;
    }

     if (data_.min_time > 0)
    {
        n++;
        m++;
    }

    if (data_.max_c3 > 0)
    {
        n++;
        m++;
    }

    // initialize optimizer
    data_.eps = eps;

    std::vector<double> tol(m, eps_t);

    auto [lower_bounds, upper_bounds] = bounds_nbody_model();

    double minf;
    opt_n_ = nlopt::opt("LD_SLSQP", n);
    opt_n_.set_lower_bounds(lower_bounds);
    opt_n_.set_upper_bounds(upper_bounds);
    opt_n_.set_min_objective(objective_nbody_model, &data_);
    opt_n_.add_equality_mconstraint(constraints_nbody_model, &data_, tol);
    opt_n_.set_maxeval(max_eval);
    opt_n_.set_ftol_abs(eps_f);
    opt_n_.optimize(xn_, minf);
}

// creates output for conic model
FlightPlan::Result FlightPlan::output_nbody_result(double eps)
{
    Result result;

    int n = 34 * (data_.num_bodies - 1) + 7 * data_.num_moon_flybys + 10;
    int m = 22 * (data_.num_bodies - 1) + 7 * data_.num_moon_flybys + 6;

    if (data_.start_time > 0)
    {
        m++;
    }

    if (data_.min_time > 0)
    {
        n++;
        m++;
    }

    if (data_.max_c3 > 0)
    {
        n++;
        m++;
    }

    data_.eps = eps;

    result.nlopt_constraints.resize(m);
    result.r.resize(3);
    result.v.resize(3);
    result.rb.resize(data_.num_bodies, std::vector<std::vector<double>>(3));
    result.vb.resize(data_.num_bodies, std::vector<std::vector<double>>(3));

    try
    {
        result.nlopt_code = opt_n_.last_optimize_result();
        result.nlopt_value = opt_n_.last_optimum_value();
        result.nlopt_num_evals = opt_n_.get_numevals();
    }
    catch (const std::exception& e)
    {

    }

    result.nlopt_solution = xn_;

    constraints_nbody_model(m, result.nlopt_constraints.data(), n, xn_.data(), NULL, &data_);

    result.time_scale = data_.ephemeris->get_time_scale();
    result.distance_scale = data_.ephemeris->get_distance_scale();
    result.velocity_scale = data_.ephemeris->get_velocity_scale();

    result.sequence = data_.sequence;
    result.bodies = data_.ephemeris->get_bodies();
    result.mu = data_.ephemeris->get_mu();

    // initialize data
    int index_t0 = 16 * data_.num_phases + 5 * data_.num_moon_flybys + 4;
    int num_dt = 6 * data_.num_phases + data_.num_moon_flybys;
    int num_dv = 3 * data_.num_phases;

    int num_phases = data_.num_phases;
    int index_dt = index_t0 + 1;
    int index_dv = index_dt + num_dt;
    int index_dvn = index_dv + 3 * num_dv;
    int index_sl = index_dvn + num_dv;

    std::vector<double> t(num_dt + 1);

    t[0] = xn_[index_t0];
    for (int i = 0; i < num_dt; i++)
    {
        t[i + 1] = t[i] + xn_[index_dt + i];
    }

    // //
    // // calculate initial conditions
    // //

    const double* t_ptr = t.data();
    const double* x_ptr = xn_.data();
    std::vector<PhaseData> phase_data(num_phases);

    // times

    for (int i = 0; i < num_phases; i++)
    {
        init_phase(phase_data[i], i, t_ptr, x_ptr, &data_);
    }

    auto output_integrate = [this, &result](IntegrationData& integration_data, int phase, int leg, int direction, double d_scale, bool ref, int bref, int offset)
    {
        double* yi = integration_data.yi.data();
        double ti = integration_data.ti;
        double t_scale_output = 1.0;
        double d_scale_output = 1.0;
        double v_scale_output = 1.0;

        double tf;
        if (direction == 1)
        {
            tf = integration_data.tf;
        }
        else
        {
            tf = integration_data.tb;
        }

        Equation equation;
        astrodynamics::NBodyEphemerisParams p;
        astrodynamics::NBodyRefEphemerisParams pref;

        void* params;

        if (ref)
        {
            pref = astrodynamics::reference_frame_params(data_.ephemeris, bref, d_scale);
            params = &pref;
            t_scale_output = pref.reference_time_scale;
            d_scale_output = pref.reference_distance_scale;
            v_scale_output = pref.reference_velocity_scale;
            equation = Equation(astrodynamics::n_body_df_ephemeris_ref, 6, astrodynamics::state_to_reference_frame(pref, yi).data(), ti / t_scale_output, data_.eps, data_.eps, params);
        }
        else
        {
            p = astrodynamics::NBodyEphemerisParams({ data_.ephemeris });
            params = &p;

            if (bref != -1)
            {
                equation = Equation(astrodynamics::n_body_df_ephemeris, 6, astrodynamics::state_add_body(data_.ephemeris, yi, bref, ti).data(), ti, data_.eps, data_.eps, params);
            }
            else
            {
                equation = Equation(astrodynamics::n_body_df_ephemeris, 6, yi, ti, data_.eps, data_.eps, params);
            }
        }

        int steps = 50;

        
        equation.stepn(tf / t_scale_output);
        std::vector<double> yf(6);
        for (int i = 0; i < 6; i++)
        {
            yf[i] = equation.get_y(i);
        }

        if (direction == 1)
        {
            integration_data.yff = yf;
        }
        else
        {
            integration_data.yfb = yf;
        }

        if (ti > tf)
        {
            double temp = tf;
            tf = ti;
            ti = temp;
        }

        for (int i = 0; i <= steps; i++)
        {
            double tout = (tf - ti) * static_cast<double>(i) / steps + ti;
            equation.stepn(tout / t_scale_output);

            Jdate jd = data_.ephemeris->get_jdate(tout);
            result.julian_time.insert(result.julian_time.end() + (steps + 1) * offset, jd.get_julian_date());
            result.kerbal_time.insert(result.kerbal_time.end() + (steps + 1) * offset, jd.get_kerbal_time());
            result.phase.insert(result.phase.end() + (steps + 1) * offset, phase);
            result.leg.insert(result.leg.end() + (steps + 1) * offset, leg);

            if (leg < 6)
            {
                result.body.insert(result.body.end() + (steps + 1) * offset, data_.sequence[phase]);
            }
            else
            {
                result.body.insert(result.body.end() + (steps + 1) * offset, data_.sequence[phase + 1]);
            }

            if (ref)
            {
                auto [rb, vb] = data_.ephemeris->get_position_velocity(bref, tout);

                result.r[0].insert(result.r[0].end() + (steps + 1) * offset, data_.ephemeris->get_distance_scale() * (equation.get_y(0) * d_scale_output + rb(0)));
                result.r[1].insert(result.r[1].end() + (steps + 1) * offset, data_.ephemeris->get_distance_scale() * (equation.get_y(1) * d_scale_output + rb(1)));
                result.r[2].insert(result.r[2].end() + (steps + 1) * offset, data_.ephemeris->get_distance_scale() * (equation.get_y(2) * d_scale_output + rb(2)));
                result.v[0].insert(result.v[0].end() + (steps + 1) * offset, data_.ephemeris->get_velocity_scale() * (equation.get_y(3) * v_scale_output + vb(0)));
                result.v[1].insert(result.v[1].end() + (steps + 1) * offset, data_.ephemeris->get_velocity_scale() * (equation.get_y(4) * v_scale_output + vb(1)));
                result.v[2].insert(result.v[2].end() + (steps + 1) * offset, data_.ephemeris->get_velocity_scale() * (equation.get_y(5) * v_scale_output + vb(2)));
            }
            else
            {
                result.r[0].insert(result.r[0].end() + (steps + 1) * offset, data_.ephemeris->get_distance_scale() * (equation.get_y(0)));
                result.r[1].insert(result.r[1].end() + (steps + 1) * offset, data_.ephemeris->get_distance_scale() * (equation.get_y(1)));
                result.r[2].insert(result.r[2].end() + (steps + 1) * offset, data_.ephemeris->get_distance_scale() * (equation.get_y(2)));
                result.v[0].insert(result.v[0].end() + (steps + 1) * offset, data_.ephemeris->get_velocity_scale() * (equation.get_y(3)));
                result.v[1].insert(result.v[1].end() + (steps + 1) * offset, data_.ephemeris->get_velocity_scale() * (equation.get_y(4)));
                result.v[2].insert(result.v[2].end() + (steps + 1) * offset, data_.ephemeris->get_velocity_scale() * (equation.get_y(5)));
            }

            for (int j = 0; j < data_.num_bodies; j++)
            {
                int b = data_.sequence[j];
                auto [rb, vb] = data_.ephemeris->get_position_velocity(b, tout);
                result.rb[j][0].insert(result.rb[j][0].end() + (steps + 1) * offset, data_.ephemeris->get_distance_scale() * (rb(0)));
                result.rb[j][1].insert(result.rb[j][1].end() + (steps + 1) * offset, data_.ephemeris->get_distance_scale() * (rb(1)));
                result.rb[j][2].insert(result.rb[j][2].end() + (steps + 1) * offset, data_.ephemeris->get_distance_scale() * (rb(2)));
                result.vb[j][0].insert(result.vb[j][0].end() + (steps + 1) * offset, data_.ephemeris->get_velocity_scale() * (vb(0)));
                result.vb[j][1].insert(result.vb[j][1].end() + (steps + 1) * offset, data_.ephemeris->get_velocity_scale() * (vb(1)));
                result.vb[j][2].insert(result.vb[j][2].end() + (steps + 1) * offset, data_.ephemeris->get_velocity_scale() * (vb(2)));
            }
        }
    };

    // phase 0, 1, 2, 3, 4
    // body (start or end) 0, 1
    // leg:
    // 0: body to moon or soi
    // 1: moon to body
    // 2: moon to soi
    // 3 soi to moon or body
    // 4: soi to dv
    // 5: dv to mp

    for (int i = 0; i < num_phases; i++)
    {
        int b_forward = phase_data[i].b_forward;
        int b_backward = phase_data[i].b_backward;
        double d_scale_forward = phase_data[i].d_scale_forward;
        double d_scale_backward = phase_data[i].d_scale_backward;

        // body forward
        output_integrate(phase_data[i].body_forward, i, 0, 1, d_scale_forward, true, b_forward, 0);

        // moon forward
        if (data_.sequence[i] == data_.index_home_body)
        {
            output_integrate(phase_data[i].moon_forward, i, 1, -1, d_scale_forward, true, b_forward, 0);
            output_integrate(phase_data[i].moon_forward, i, 2, 1, d_scale_forward, true, b_forward, 0);
        }

        // soi forward
        output_integrate(phase_data[i].soi_forward, i, 3, -1, d_scale_forward, true, b_forward, 0);
        output_integrate(phase_data[i].soi_forward, i, 4, 1, d_scale_forward, false, b_forward, 0);

        // dv1
        phase_data[i].dv1.yi = add_dv(phase_data[i].soi_forward.yff.data(), &xn_[index_dv + 3 * (3 * i)], xn_[index_dvn + 3 * i]);
        output_integrate(phase_data[i].dv1, i, 5, 1, 0, false, -1, 0);

        // soi backward
        output_integrate(phase_data[i].soi_backward, i, 7, -1, d_scale_backward, false, b_backward, 0);
        output_integrate(phase_data[i].soi_backward, i, 8, 1, d_scale_backward, true, b_backward, 0);

        // dv3
        phase_data[i].dv3.yi = add_dv(phase_data[i].soi_backward.yfb.data(), &xn_[index_dv + 3 * (3 * (i + 1) - 1)], -xn_[index_dvn + (3 * (i + 1) - 1)]);
        output_integrate(phase_data[i].dv3, i, 6, -1, 0, false, -1, -2);

        // moon backward
        if (data_.sequence[i + 1] == data_.index_home_body)
        {
            output_integrate(phase_data[i].moon_backward, i, 9, -1, d_scale_backward, true, b_backward, 0);
            output_integrate(phase_data[i].moon_backward, i, 10, 1, d_scale_backward, true, b_backward, 0);
        }

        // body backward
        output_integrate(phase_data[i].body_backward, i, 11, -1, d_scale_backward, true, b_backward, 0);
    }

    return result;
}


// set bounds for decisions variables for numerical model
std::tuple<std::vector<double>, std::vector<double>> FlightPlan::bounds_nbody_model()
{
    std::vector<double> lower_bounds;
    std::vector<double> upper_bounds;

    int index_t0 = 16 * data_.num_phases + 5 * data_.num_moon_flybys + 4;
    int num_dt = 6 * data_.num_phases + data_.num_moon_flybys;
    int num_dv = 3 * data_.num_phases;

    double orbital_radius_launch = data_.radius_min.front();
    double orbital_radius_arrival = data_.radius_min.back();

    double pi = astrodynamics::pi;
    double ms = 1.0 / data_.ephemeris->get_velocity_scale();
    double d = 1.0 / data_.ephemeris->get_distance_scale();
    double min_dt = 0.01 * day_;
    double max_dt = 10 * year_;
    double min_dv = -5000 * ms;
    double max_dv = 5000 * ms;

    // position and velocity bounds

    // start planet and soi bounds
    lower_bounds.push_back(-2.0 * pi);
    lower_bounds.push_back(-2.0 * pi);
    lower_bounds.push_back(sqrt(2.0 * data_.ephemeris->get_mu(data_.sequence[0]) / orbital_radius_launch));
    lower_bounds.push_back(-2.0 * pi);
    lower_bounds.push_back(-2.0 * pi);

    lower_bounds.push_back(-2.0 * pi);
    lower_bounds.push_back(-2.0 * pi);
    lower_bounds.push_back(ms);
    lower_bounds.push_back(-2.0 * pi);
    lower_bounds.push_back(-2.0 * pi);

    upper_bounds.push_back(2.0 * pi);
    upper_bounds.push_back(2.0 * pi);
    upper_bounds.push_back(sqrt(10.0 * data_.ephemeris->get_mu(data_.sequence[0]) / orbital_radius_launch));
    upper_bounds.push_back(2.0 * pi);
    upper_bounds.push_back(2.0 * pi);

    upper_bounds.push_back(2.0 * pi);
    upper_bounds.push_back(2.0 * pi);
    upper_bounds.push_back(sqrt(10.0 * data_.ephemeris->get_mu(data_.sequence[0]) / orbital_radius_launch));
    upper_bounds.push_back(2.0 * pi);
    upper_bounds.push_back(2.0 * pi);

    // bounds at moon orbit radius
    if (data_.sequence[0] == data_.index_home_body)
    {
        lower_bounds.push_back(-2.0 * pi);
        lower_bounds.push_back(-2.0 * pi);
        lower_bounds.push_back(ms);
        lower_bounds.push_back(-2.0 * pi);
        lower_bounds.push_back(-2.0 * pi);

        upper_bounds.push_back(2.0 * pi);
        upper_bounds.push_back(2.0 * pi);
        upper_bounds.push_back(sqrt(10.0 * data_.ephemeris->get_mu(data_.sequence[0]) / orbital_radius_launch));
        upper_bounds.push_back(2.0 * pi);
        upper_bounds.push_back(2.0 * pi);
    }

    // intermediate planet bounds
    for (int i = 1; i < data_.num_bodies - 1; i++)
    {
        // bounds at moon orbit radius
        if (data_.sequence[i] == data_.index_home_body)
        {
            lower_bounds.push_back(-2.0 * pi);
            lower_bounds.push_back(-2.0 * pi);
            lower_bounds.push_back(ms);
            lower_bounds.push_back(-2.0 * pi);
            lower_bounds.push_back(-2.0 * pi);

            upper_bounds.push_back(2.0 * pi);
            upper_bounds.push_back(2.0 * pi);
            upper_bounds.push_back(sqrt(10.0 * data_.ephemeris->get_mu(data_.sequence[i]) / data_.radius_min[i]));
            upper_bounds.push_back(2.0 * pi);
            upper_bounds.push_back(2.0 * pi);
        }

            // bounds at soi, body, soi
            lower_bounds.push_back(-2.0 * pi);
            lower_bounds.push_back(-2.0 * pi);
            lower_bounds.push_back(ms);
            lower_bounds.push_back(-2.0 * pi);
            lower_bounds.push_back(-2.0 * pi);

            lower_bounds.push_back(data_.radius_min[i] - 1e-10);
            lower_bounds.push_back(-2.0 * pi);
            lower_bounds.push_back(-2.0 * pi);
            lower_bounds.push_back(ms);
            lower_bounds.push_back(-2.0 * pi);
            lower_bounds.push_back(-2.0 * pi);

            lower_bounds.push_back(-2.0 * pi);
            lower_bounds.push_back(-2.0 * pi);
            lower_bounds.push_back(ms);
            lower_bounds.push_back(-2.0 * pi);
            lower_bounds.push_back(-2.0 * pi);

            upper_bounds.push_back(2.0 * pi);
            upper_bounds.push_back(2.0 * pi);
            upper_bounds.push_back(sqrt(10.0 * data_.ephemeris->get_mu(data_.sequence[i]) / data_.radius_min[i]));
            upper_bounds.push_back(2.0 * pi);
            upper_bounds.push_back(2.0 * pi);

            upper_bounds.push_back(0.50 * data_.radius_soi[i]);
            upper_bounds.push_back(2.0 * pi);
            upper_bounds.push_back(2.0 * pi);
            upper_bounds.push_back(sqrt(10.0 * data_.ephemeris->get_mu(data_.sequence[i]) / data_.radius_min[i]));
            upper_bounds.push_back(2.0 * pi);
            upper_bounds.push_back(2.0 * pi);

            upper_bounds.push_back(2.0 * pi);
            upper_bounds.push_back(2.0 * pi);
            upper_bounds.push_back(sqrt(10.0 * data_.ephemeris->get_mu(data_.sequence[i]) / data_.radius_min[i]));
            upper_bounds.push_back(2.0 * pi);
            upper_bounds.push_back(2.0 * pi);

        // bounds at moon orbit radius
        if (data_.sequence[i] == data_.index_home_body)
        {
            lower_bounds.push_back(-2.0 * pi);
            lower_bounds.push_back(-2.0 * pi);
            lower_bounds.push_back(ms);
            lower_bounds.push_back(-2.0 * pi);
            lower_bounds.push_back(-2.0 * pi);

            upper_bounds.push_back(2.0 * pi);
            upper_bounds.push_back(2.0 * pi);
            upper_bounds.push_back(sqrt(10.0 * data_.ephemeris->get_mu(data_.sequence[i]) / data_.radius_min[i]));
            upper_bounds.push_back(2.0 * pi);
            upper_bounds.push_back(2.0 * pi);
        }      
    }

    // bounds at moon orbit radius
    if (data_.sequence[data_.num_bodies - 1] == data_.index_home_body)
    {
        lower_bounds.push_back(-2.0 * pi);
        lower_bounds.push_back(-2.0 * pi);
        lower_bounds.push_back(ms);
        lower_bounds.push_back(-2.0 * pi);
        lower_bounds.push_back(-2.0 * pi);

        upper_bounds.push_back(2.0 * pi);
        upper_bounds.push_back(2.0 * pi);
        upper_bounds.push_back(sqrt(10.0 * data_.ephemeris->get_mu(data_.sequence[data_.num_bodies - 1]) / orbital_radius_arrival));
        upper_bounds.push_back(2.0 * pi);
        upper_bounds.push_back(2.0 * pi);
    }

    // bounds at end planet and soi boundary
    lower_bounds.push_back(-2.0 * pi);
    lower_bounds.push_back(-2.0 * pi);
    lower_bounds.push_back(ms);
    lower_bounds.push_back(-2.0 * pi);
    lower_bounds.push_back(-2.0 * pi);

    lower_bounds.push_back(-2.0 * pi);
    lower_bounds.push_back(-2.0 * pi);
    lower_bounds.push_back(sqrt(2.0 * data_.ephemeris->get_mu(data_.sequence[data_.num_bodies - 1]) / orbital_radius_arrival));
    lower_bounds.push_back(-2.0 * pi);
    lower_bounds.push_back(-2.0 * pi);

    upper_bounds.push_back(2.0 * pi);
    upper_bounds.push_back(2.0 * pi);
    upper_bounds.push_back(sqrt(10.0 * data_.ephemeris->get_mu(data_.sequence[data_.num_bodies - 1]) / orbital_radius_arrival));
    upper_bounds.push_back(2.0 * pi);
    upper_bounds.push_back(2.0 * pi);

    upper_bounds.push_back(2.0 * pi);
    upper_bounds.push_back(2.0 * pi);
    upper_bounds.push_back(sqrt(10.0 * data_.ephemeris->get_mu(data_.sequence[data_.num_bodies - 1]) / orbital_radius_arrival));
    upper_bounds.push_back(2.0 * pi);
    upper_bounds.push_back(2.0 * pi);

    // start time bounds
    if (data_.min_start_time > 0)
    {
        lower_bounds.push_back(data_.min_start_time);
    }
    else if (xn_[index_t0] > year_)
    {
        lower_bounds.push_back(xn_[index_t0] - year_);
    }
    else
    {
        lower_bounds.push_back(0.0);
    }

    upper_bounds.push_back(xn_[index_t0] + year_);

    // delta time bounds
    for (int i = 0; i < num_dt; i++)
    {
        lower_bounds.push_back(min_dt);
        upper_bounds.push_back(max_dt);
    }

    // dsm unit vector bounds
    for (int i = 0; i < 3 * num_dv; i++)
    {
        lower_bounds.push_back(-1.0);
        upper_bounds.push_back(1.0);
    }

    // dsm magnitude bounds
    for (int i = 0; i < num_dv; i++)
    {
        lower_bounds.push_back(0);
        upper_bounds.push_back(max_dv);
    }

    //
    // slack variable bounds
    //

    // lunar distance bounds
    for (int i = 0; i < data_.num_moon_flybys; i++)
    {
        lower_bounds.push_back(0.0);
        upper_bounds.push_back(HUGE_VAL);
    }

    // inclination bounds
    lower_bounds.push_back(0.0);
    lower_bounds.push_back(0.0);
    lower_bounds.push_back(0.0);
    lower_bounds.push_back(0.0);

    upper_bounds.push_back(10.0);
    upper_bounds.push_back(10.0);
    upper_bounds.push_back(10.0);
    upper_bounds.push_back(10.0);

    // max time of flight bounds
    lower_bounds.push_back(0.0);
    upper_bounds.push_back(HUGE_VAL);

    // min time of flight bounds
    if (data_.min_time > 0)
    {
        lower_bounds.push_back(0.0);
        upper_bounds.push_back(HUGE_VAL);
    }

    // c3 bounds
    if (data_.max_c3 > 0)
    {
        lower_bounds.push_back(0.0);
        upper_bounds.push_back(HUGE_VAL);
    }

    return { lower_bounds, upper_bounds };
}


// calculate constraints and derivatives of numerical model for nlopt
void FlightPlan::constraints_nbody_model(unsigned m, double *result, unsigned n, const double* x, double* grad, void* f_data)
{
    try
    {
        //
        // parse inputs
        //

        // convert void pointer to data pointer
        FlightPlanData* data = reinterpret_cast<FlightPlanData*>(f_data);

        // calculate number of phases, time intervals, and delta velocity maneuvers
        int num_phases = data->num_bodies - 1;
        int num_dt = 6 * num_phases + data->num_moon_flybys;
        int num_dv = 3 * num_phases;
        
        // calculate indicies for inputs within x
        int index_t0 = 16 * num_phases + 5 * data->num_moon_flybys + 4;
        int index_dt = index_t0 + 1;
        int index_dv_unit = index_dt + num_dt;
        int index_dv_norm = index_dv_unit + 3 * num_dv;
        int index_sl = index_dv_norm + num_dv;

        // initialize time vector
        std::vector<double> t(num_dt + 1);

        t[0] = x[index_t0];
        for (int i = 0; i < num_dt; i++)
        {
            t[i + 1] = t[i] + x[index_dt + i];
        }

        //
        // calculate initial conditions
        //

        const double* t_ptr = t.data();
        const double* x_ptr = x;
        std::vector<PhaseData> phase_data(num_phases);
        Eigen::MatrixXd mgrad = Eigen::MatrixXd::Zero(m, n);

        for (int i = 0; i < num_phases; i++)
        {
            init_phase(phase_data[i], i, t_ptr, x_ptr, data);
        }

        //
        // integrate
        //

        for (int i = 0; i < num_phases; i++)
        {
            int no_body = -1;
            bool forward = true;
            bool backward = false;

            //
            // forward direction
            //

            int b = phase_data[i].b_forward;
            double d_scale = phase_data[i].d_scale_forward;

            // forward integration from body in body centered reference frame
            integrate(phase_data[i].body_forward, data->ephemeris, forward, d_scale, true, b, data->eps, grad);

            // backward/forward integration from moon in body centered reference frame
            if (data->sequence[i] == data->index_home_body)
            {
                integrate(phase_data[i].moon_forward, data->ephemeris, backward, d_scale, true, b, data->eps, grad);
                integrate(phase_data[i].moon_forward, data->ephemeris, forward, d_scale, true, b, data->eps, grad);
            }

            // backward/forward integration from sphere of influence, backward integration in body
            // centered reference frame, forward integration in sun centered reference frame
            integrate(phase_data[i].soi_forward, data->ephemeris, backward, d_scale, true, b, data->eps, grad);
            integrate(phase_data[i].soi_forward, data->ephemeris, forward, d_scale, false, b, data->eps, grad);

            // forward integration from first delta velocity maneuever, delta velocity added to
            // output from forward integration output from sphere of influence
            phase_data[i].dv1.yi = add_dv(phase_data[i].soi_forward.yff.data(), &x[index_dv_unit + 9 * i], x[index_dv_norm + 3 * i]);
            integrate(phase_data[i].dv1, data->ephemeris, forward, d_scale, false, no_body, data->eps, grad);

            //
            // backward direction
            //

            b = phase_data[i].b_backward;
            d_scale = phase_data[i].d_scale_backward;

            // backward/forward integration from sphere of influence, backward integration in sun
            // centered reference frame, forward integration in body centered reference frame
            integrate(phase_data[i].soi_backward, data->ephemeris, backward, d_scale, false, b, data->eps, grad);
            integrate(phase_data[i].soi_backward, data->ephemeris, forward, d_scale, true, b, data->eps, grad);

            // backward integration from third delta velocity maneuever, delta velocity added to
            // output from backward integration output from sphere of influence
            phase_data[i].dv3.yi = add_dv(phase_data[i].soi_backward.yfb.data(), &x[index_dv_unit + 9 * i + 6], -x[index_dv_norm + 3 * i + 2]);
            integrate(phase_data[i].dv3, data->ephemeris, backward, d_scale, false, no_body, data->eps, grad);
            
            // backward/forward integration from moon in body centered reference frame
            if (data->sequence[i + 1] == data->index_home_body)
            {
                integrate(phase_data[i].moon_backward, data->ephemeris, backward, d_scale, true, b, data->eps, grad);
                integrate(phase_data[i].moon_backward, data->ephemeris, forward, d_scale, true, b, data->eps, grad);
            }

            // backward integration from body in body centered reference frame
            integrate(phase_data[i].body_backward, data->ephemeris, backward, d_scale, true, b, data->eps, grad);
        }
        
        //
        // calculate result
        //

        double* result_ptr = result;

        // match point constraints
        for (int i = 0; i < num_phases; i++)
        {
            if (data->sequence[i] == data->index_home_body)
            {
                for (int j = 0; j < 6; j++)
                {
                    *result_ptr++ = phase_data[i].moon_forward.yfb[j] - phase_data[i].body_forward.yff[j];
                }

                for (int j = 0; j < 6; j++)
                {
                    *result_ptr++ = phase_data[i].soi_forward.yfb[j] - phase_data[i].moon_forward.yff[j];
                }
            }
            else
            {
                for (int j = 0; j < 6; j++)
                {
                    *result_ptr++ = phase_data[i].soi_forward.yfb[j] - phase_data[i].body_forward.yff[j];
                }
            }

            *result_ptr++ =  phase_data[i].dv3.yfb[0] - phase_data[i].dv1.yff[0];
            *result_ptr++ =  phase_data[i].dv3.yfb[1] - phase_data[i].dv1.yff[1];
            *result_ptr++ =  phase_data[i].dv3.yfb[2] - phase_data[i].dv1.yff[2];
            *result_ptr++ =  phase_data[i].dv3.yfb[3] - phase_data[i].dv1.yff[3] - x[index_dv_norm + 3 * i + 1] * x[index_dv_unit + 9 * i + 3];
            *result_ptr++ =  phase_data[i].dv3.yfb[4] - phase_data[i].dv1.yff[4] - x[index_dv_norm + 3 * i + 1] * x[index_dv_unit + 9 * i + 4];
            *result_ptr++ =  phase_data[i].dv3.yfb[5] - phase_data[i].dv1.yff[5] - x[index_dv_norm + 3 * i + 1] * x[index_dv_unit + 9 * i + 5];
            
            if (data->sequence[i + 1] == data->index_home_body)
            {
                for (int j = 0; j < 6; j++)
                {
                    *result_ptr++ = phase_data[i].moon_backward.yfb[j] - phase_data[i].soi_backward.yff[j];
                }

                for (int j = 0; j < 6; j++)
                {
                    *result_ptr++ = phase_data[i].body_backward.yfb[j] - phase_data[i].moon_backward.yff[j];
                }
            }
            else
            {
                for (int j = 0; j < 6; j++)
                {
                    *result_ptr++ = phase_data[i].body_backward.yfb[j] - phase_data[i].soi_backward.yff[j];
                }
            }
        }

        // periapsis constraint
        for (int i = 0; i < data->num_bodies; i++)
        {
            if (i < data->num_bodies - 1)
            {
                *result_ptr++ = phase_data[i].body_forward.yi[0] * phase_data[i].body_forward.yi[3] + phase_data[i].body_forward.yi[1] * phase_data[i].body_forward.yi[4] + phase_data[i].body_forward.yi[2] * phase_data[i].body_forward.yi[5];
            }
            else
            {
                *result_ptr++ = phase_data[i - 1].body_backward.yi[0] * phase_data[i - 1].body_backward.yi[3] + phase_data[i - 1].body_backward.yi[1] * phase_data[i - 1].body_backward.yi[4] + phase_data[i - 1].body_backward.yi[2] * phase_data[i - 1].body_backward.yi[5];
            } 
        }

        // dv unit vector constraint
        for (int i = 0; i < 3 * num_phases; i++)
        {	
            double ux = x[index_dv_unit + 3 * i + 0];
            double uy = x[index_dv_unit + 3 * i + 1];
            double uz = x[index_dv_unit + 3 * i + 2];

            *result_ptr++ = ux * ux + uy * uy + uz * uz - 1.0;
        }

        // lunar constraints
        for (int i = 0; i < num_phases; i++)
        {
            if (data->sequence[i] == data->index_home_body)
            {
                Eigen::Vector3d rm = data->ephemeris->get_position(data->index_home_moon, phase_data[i].moon_forward.ti) - data->ephemeris->get_position(data->index_home_body, phase_data[i].moon_forward.ti);
                double drmx = rm(0) - phase_data[i].moon_forward.yi[0];
                double drmy = rm(1) - phase_data[i].moon_forward.yi[1];
                double drmz = rm(2) - phase_data[i].moon_forward.yi[2];

                *result_ptr++ = drmx * drmx + drmy * drmy + drmz * drmz - data->min_moon_distance_squared - x[index_sl];
                index_sl++;
            }
            
            if (data->sequence[i + 1] == data->index_home_body)
            {
                Eigen::Vector3d rm = data->ephemeris->get_position(data->index_home_moon, phase_data[i].moon_backward.ti) - data->ephemeris->get_position(data->index_home_body, phase_data[i].moon_backward.ti);
                double drmx = rm(0) - phase_data[i].moon_backward.yi[0];
                double drmy = rm(1) - phase_data[i].moon_backward.yi[1];
                double drmz = rm(2) - phase_data[i].moon_backward.yi[2];
                *result_ptr++ = drmx * drmx + drmy * drmy + drmz * drmz - data->min_moon_distance_squared - x[index_sl];
                index_sl++;
            }
        }

        // ra, dec, vrho, vra, vdec
        //  0    1     2    3     4
        // -5   -4    -3   -2    -1

        // inclination constraints
        double cosi_launch = cos(x[1]) * cos(x[4]) * sin(x[3] - x[0]);
        double cosi_arrival = cos(x[index_t0 - 4]) * cos(x[index_t0 - 1]) * sin(x[index_t0 - 2] - x[index_t0 - 5]);

        *result_ptr++ = cosi_launch - cos(data->min_inclination_launch) + x[index_sl];
        index_sl++;
        *result_ptr++ = cosi_launch - cos(data->max_inclination_launch) - x[index_sl];
        index_sl++;

        *result_ptr++ = cosi_arrival - cos(data->min_inclination_arrival) + x[index_sl];
        index_sl++;
        *result_ptr++ = cosi_arrival - cos(data->max_inclination_arrival) - x[index_sl];
        index_sl++;

        // max time of flight constraint
        *result_ptr++ = t[t.size() - 1] - t[0] - data->max_time + x[index_sl];
        index_sl++;

        // min time of flight constraint
        if (data->min_time > 0)
        {
            *result_ptr++ = t[t.size() - 1] - t[0] - data->min_time - x[index_sl];
            index_sl++;
        }

        // c3 constraints
        if (data->max_c3 > 0)
        {
            double v_launch = phase_data[0].body_forward.xi[3];
            double c3 = v_launch * v_launch - 2 * data->ephemeris->get_mu(data->sequence[0]) / data->radius_min[0];
            *result_ptr++ = c3 - data->max_c3 + x[index_sl];
        }

        // accounts for start time constraint
        if (data->start_time > 0)
        {
            *result_ptr++ = data->start_time - t[0];
        }

        if (grad)
        {
            //
            // column indices
            //
            
            int col_index_mp = 0;
            int col_index_dt = index_dt;
            int col_index_dv_unit = index_dv_unit;
            int col_index_dv_norm = index_dv_norm;

            index_sl = index_dv_norm + num_dv;
                    
            //
            // row indices, start time constraint
            //
            
            int row_index_mp = 0;

            int row_index_tstart = m - 1;
            int row_index_c3 = m - 1;
            int row_index_tof = m - 1;
            int row_index_min_time = m - 1;

            if (data->start_time > 0)
            {
                mgrad(m - 1, index_t0) = -1.0;
                row_index_c3--;
                row_index_tof--;
                row_index_min_time--;
            }

            if (data->max_c3 > 0)
            {
                mgrad(row_index_c3, 2) = 2 * phase_data[0].body_forward.xi[3];
                row_index_tof--;
                row_index_min_time--;
            }

            if (data->min_time > 0)
            {
                row_index_tof--;
            }

            int row_index_inc = row_index_tof - 4;
            int row_index_moon = row_index_inc - data->num_moon_flybys;
            int row_index_dv_unit = row_index_moon - 3 * num_phases ;
            int row_index_periapsis = row_index_dv_unit - data->num_bodies;

            //
            // calculation of partial derivatives 
            //
            
            for (int i = 0; i < num_phases; i++)
            {
                // match point derivatives between body and moon or soi and periapsis derivatives with respect to initial state at body, if first phase radius is not a decision variable
                if (i == 0)
                {
                    mgrad.block<6, 5>(row_index_mp, col_index_mp) = -spherical_stm(phase_data[i].body_forward, 1, data->d_scale[i], data->v_scale[i]).block<6, 5>(0, 1);
                    mgrad.block<1, 5>(row_index_periapsis, col_index_mp) = spherical_derivative_r_dot_v(phase_data[i].body_forward.xi).block<5, 1>(1, 0);
                    row_index_periapsis++;
                    col_index_mp += 5;
                }
                else
                {
                    mgrad.block<6, 6>(row_index_mp, col_index_mp) = -spherical_stm(phase_data[i].body_forward, 1, data->d_scale[i], data->v_scale[i]);
                    mgrad.block<1, 6>(row_index_periapsis, col_index_mp) = spherical_derivative_r_dot_v(phase_data[i].body_forward.xi);
                    row_index_periapsis++;
                    col_index_mp += 6;
                }
                
                // match point derivatives with respect to initial state and time within sphere of influence
                // if body is home body include lunar match points
                if (data->sequence[i] == data->index_home_body)
                {
                    // lunar distance constraint
                    double ra = phase_data[i].moon_forward.xi[1];
                    double dec = phase_data[i].moon_forward.xi[2];

                    Eigen::Vector3d rm = data->ephemeris->get_position(data->index_home_moon, phase_data[i].moon_forward.ti) - data->ephemeris->get_position(data->index_home_body, phase_data[i].moon_forward.ti);
                    Eigen::Vector3d vm = data->ephemeris->get_velocity(data->index_home_moon, phase_data[i].moon_forward.ti) - data->ephemeris->get_velocity(data->index_home_body, phase_data[i].moon_forward.ti);

                    mgrad(row_index_moon, col_index_mp) = 2 * data->moon_mean_distance * cos(dec) * (rm(0) * sin(ra) - rm(2) * cos(ra));
                    mgrad(row_index_moon, col_index_mp + 1) = 2 * data->moon_mean_distance * (rm(0) * cos(ra) * sin(dec) - rm(1) * cos(dec) + rm(2) * sin(ra) * sin(dec));;
                    mgrad(row_index_moon, index_sl) = -1; 
                    index_sl++;

                    // match point derivative between body and moon with respect to body time
                    Eigen::Matrix<double, 6, 1> dmp_body_moon_dtau_body = (
                        Eigen::Map<Eigen::Matrix<double, 6, 1>>(&phase_data[i].moon_forward.yfb[42]) - 
                        Eigen::Map<Eigen::Matrix<double, 6, 1>>(&phase_data[i].body_forward.yff[42])) / data->t_scale[i];
                    
                    for (int j = 0; j < col_index_dt - index_t0; j++)
                    {
                        mgrad.block(row_index_mp, col_index_dt - 1 - j, 6, 1) = dmp_body_moon_dtau_body;
                    }
                    
                    // match point derivative between body and moon with respect to moon time
                    mgrad.block(row_index_mp, col_index_dt, 6, 1) = (
                        Eigen::Map<Eigen::Matrix<double, 6, 1>>(&phase_data[i].moon_forward.yfb[42]) - 0.5 * (
                        Eigen::Map<Eigen::Matrix<double, 6, 1>>(&phase_data[i].body_forward.ypf[0]) +
                        Eigen::Map<Eigen::Matrix<double, 6, 1>>(&phase_data[i].moon_forward.ypb[0]))) / data->t_scale[i];          
                    col_index_dt++;
                
                    // match point derivatives between body and moon and moon and soi with respect to initial state at moon		
                    mgrad.block<6, 5>(row_index_mp, col_index_mp) = spherical_stm(phase_data[i].moon_forward, -1, data->d_scale[i], data->v_scale[i]).block<6, 5>(0, 1);;
                    row_index_mp += 6;

                    mgrad.block<6, 5>(row_index_mp, col_index_mp) = -spherical_stm(phase_data[i].moon_forward, 1, data->d_scale[i], data->v_scale[i]).block<6, 5>(0, 1);
                    col_index_mp += 5;
                    
                    // match point derivative between moon and soi with respect to moon time and lunar distance constraint with respect to moon time
                    Eigen::Matrix<double, 6, 1> dmp_moon_soi_dtau_moon = (
                        Eigen::Map<Eigen::Matrix<double, 6, 1>>(&phase_data[i].soi_forward.yfb[42]) -
                        Eigen::Map<Eigen::Matrix<double, 6, 1>>(&phase_data[i].moon_forward.yff[42])) / data->t_scale[i];

                    for (int j = 0; j < col_index_dt - index_t0; j++)
                    {
                        mgrad.block(row_index_mp, col_index_dt - 1 - j, 6, 1) = dmp_moon_soi_dtau_moon;
                        mgrad(row_index_moon, col_index_dt - 1 - j) = vm(0) * 2 * (rm(0) - phase_data[i].moon_forward.yi[0]) + vm(1) * 2 * (rm(1) - phase_data[i].moon_forward.yi[1]) + vm(2) * 2 * (rm(2) - phase_data[i].moon_forward.yi[2]);
                    }

                    row_index_moon++;

                    // match point derivative between moon and soi with respect to soi time
                    mgrad.block(row_index_mp, col_index_dt, 6, 1) = (
                        Eigen::Map<Eigen::Matrix<double, 6, 1>>(&phase_data[i].soi_forward.yfb[42]) - 0.5 * (
                        Eigen::Map<Eigen::Matrix<double, 6, 1>>(&phase_data[i].moon_forward.ypf[0]) + 
                        Eigen::Map<Eigen::Matrix<double, 6, 1>>(&phase_data[i].soi_forward.ypb[0]))) / data->t_scale[i];
                    col_index_dt++;

                    // match point derivative between moon and soi with respect to initial state at soi
                    mgrad.block<6, 5>(row_index_mp, col_index_mp) = spherical_stm(phase_data[i].soi_forward, -1, data->d_scale[i], data->v_scale[i]).block<6, 5>(0, 1);
                    row_index_mp += 6;
                }
                else
                {
                    // match point derivative between body and soi with respect to body time
                    Eigen::Matrix<double, 6, 1> dmp_body_soi_dtau_body = (
                        Eigen::Map<Eigen::Matrix<double, 6, 1>>(&phase_data[i].soi_forward.yfb[42]) -
                        Eigen::Map<Eigen::Matrix<double, 6, 1>>(&phase_data[i].body_forward.yff[42])) / data->t_scale[i];
                    
                    for (int j = 0; j < col_index_dt - index_t0; j++)
                    {
                        mgrad.block(row_index_mp, col_index_dt - 1 - j, 6, 1) = dmp_body_soi_dtau_body;
                    }
                    
                    // match point derivative between body and soi with respect to soi time
                    mgrad.block(row_index_mp, col_index_dt, 6, 1) = (
                        Eigen::Map<Eigen::Matrix<double, 6, 1>>(&phase_data[i].soi_forward.yfb[42]) - 0.5 * (
                        Eigen::Map<Eigen::Matrix<double, 6, 1>>(&phase_data[i].body_forward.ypf[0]) + 
                        Eigen::Map<Eigen::Matrix<double, 6, 1>>(&phase_data[i].soi_forward.ypb[0]))) / data->t_scale[i];;
                    col_index_dt++;
                
                    // match point derivative between body and soi with respect to initial state at soi
                    mgrad.block<6, 5>(row_index_mp, col_index_mp) = spherical_stm(phase_data[i].soi_forward, -1,  data->d_scale[i],  data->v_scale[i]).block<6, 5>(0, 1);
                    row_index_mp += 6;
                }
                            
                //
                // derivatives related to change in velocity maneuvers
                //
            
                // state transition matrix for dv1 and dv3 legs
                Eigen::Matrix<double, 6, 6> stmdv1 = Eigen::Map<Eigen::Matrix<double, 6, 6>>(&phase_data[i].dv1.yff[6]);
                Eigen::Matrix<double, 6, 6> stmdv3 = Eigen::Map<Eigen::Matrix<double, 6, 6>>(&phase_data[i].dv3.yfb[6]);
                
                // change in velocity unit vectors
                Eigen::Matrix<double, 6, 1> dv1_unit, dv2_unit, dv3_unit;
                dv1_unit << 0, 0, 0, x[index_dv_unit + 9 * i + 0], x[index_dv_unit + 9 * i + 1], x[index_dv_unit + 9 * i + 2];
                dv2_unit << 0, 0, 0, x[index_dv_unit + 9 * i + 3], x[index_dv_unit + 9 * i + 4], x[index_dv_unit + 9 * i + 5];
                dv3_unit << 0, 0, 0, x[index_dv_unit + 9 * i + 6], x[index_dv_unit + 9 * i + 7], x[index_dv_unit + 9 * i + 8];
                
                // match point derivatives with respect to change in velocity unit vectors
                mgrad.block<6, 3>(row_index_mp, col_index_dv_unit) = -stmdv1.block<6, 3>(0, 3) * x[index_dv_norm + 3 * i];
                mgrad.block<3, 3>(row_index_mp + 3, col_index_dv_unit + 3) = -Eigen::Matrix3d::Identity()* x[index_dv_norm + 3 * i + 1];
                mgrad.block<6, 3>(row_index_mp, col_index_dv_unit + 6) = -stmdv3.block<6, 3>(0, 3) * x[index_dv_norm + 3 * i + 2];
                
                // match point derivatives with respect to change in velocity magnitudes
                mgrad.block<6, 1>(row_index_mp, col_index_dv_norm + 3 * i) = -stmdv1 * dv1_unit;
                mgrad.block<6, 1>(row_index_mp, col_index_dv_norm + 3 * i + 1) = -dv2_unit;
                mgrad.block<6, 1>(row_index_mp, col_index_dv_norm + 3 * i + 2) = -stmdv3 * dv3_unit;
                
                // derivative of unit vector magnitude with respect to unit vector
                mgrad.block<1, 3>(row_index_dv_unit, col_index_dv_unit) = 2 * dv1_unit.block<3, 1>(3, 0);
                mgrad.block<1, 3>(row_index_dv_unit + 1, col_index_dv_unit + 3) = 2 * dv2_unit.block<3, 1>(3, 0);
                mgrad.block<1, 3>(row_index_dv_unit + 2, col_index_dv_unit + 6) = 2 * dv3_unit.block<3, 1>(3, 0);
                    
                //
                // derivatives related to soi state
                //
                
                mgrad.block<6, 5>(row_index_mp, col_index_mp) = -stmdv1 * spherical_stm(phase_data[i].soi_forward, 1,  1,  1).block<6, 5>(0, 1);
                col_index_mp += 5;
                mgrad.block<6, 5>(row_index_mp, col_index_mp) =  stmdv3 * spherical_stm(phase_data[i].soi_backward, -1, 1, 1).block<6, 5>(0, 1);
                
                //
                // derivatives related to time
                //
                
                // planet state derivatives with respect to time at soi time
                Eigen::Vector3d vb0 =  data->ephemeris->get_velocity(data->sequence[i], phase_data[i].soi_forward.ti);
                Eigen::Vector3d ab0 =  data->ephemeris->get_acceleration(data->sequence[i], phase_data[i].soi_forward.ti);
                Eigen::Matrix<double, 6, 1> xbp0;
                xbp0 << vb0(0), vb0(1), vb0(2), ab0(0), ab0(1), ab0(2);
                
                Eigen::Vector3d vb1 = data->ephemeris->get_velocity(data->sequence[i + 1], phase_data[i].soi_backward.ti);
                Eigen::Vector3d ab1 = data->ephemeris->get_acceleration(data->sequence[i + 1], phase_data[i].soi_backward.ti);
                Eigen::Matrix<double, 6, 1> xbp1;
                xbp1 << vb1(0), vb1(1), vb1(2), ab1(0), ab1(1), ab1(2);
                
                // partial derivatives with respect to epoch time 
                Eigen::Matrix<double, 6, 1> pdmp_dtau_dv1 = Eigen::Map<Eigen::Matrix<double, 6, 1>>(&phase_data[i].dv1.yff[42]);
                Eigen::Matrix<double, 6, 1> pdmp_dtau_dv3 = Eigen::Map<Eigen::Matrix<double, 6, 1>>(&phase_data[i].dv3.yfb[42]);
                Eigen::Matrix<double, 6, 1> pdmp_dtau_soif = Eigen::Map<Eigen::Matrix<double, 6, 1>>(&phase_data[i].soi_forward.yff[42]);
                Eigen::Matrix<double, 6, 1> pdmp_dtau_soib = Eigen::Map<Eigen::Matrix<double, 6, 1>>(&phase_data[i].soi_backward.yfb[42]);
                
                // partial derivatives with respect to time
                Eigen::Matrix<double, 6, 1> dmp_dt_dv1 = Eigen::Map<Eigen::Matrix<double, 6, 1>>(&phase_data[i].dv1.ypf[0]);
                Eigen::Matrix<double, 6, 1> dmp_dt_dv3 = Eigen::Map<Eigen::Matrix<double, 6, 1>>(&phase_data[i].dv3.ypb[0])	;
                Eigen::Matrix<double, 6, 1> dmp_dt_soif = Eigen::Map<Eigen::Matrix<double, 6, 1>>(&phase_data[i].soi_forward.ypf[0]);
                Eigen::Matrix<double, 6, 1> dmp_dt_soib = Eigen::Map<Eigen::Matrix<double, 6, 1>>(&phase_data[i].soi_backward.ypb[0]);
                
                // state transition matrix for soi legs
                Eigen::Matrix<double, 6, 6> stmsoif = Eigen::Map<Eigen::Matrix<double, 6, 6>>(&phase_data[i].soi_forward.yff[6]);
                Eigen::Matrix<double, 6, 6> stmsoib = Eigen::Map<Eigen::Matrix<double, 6, 6>>(&phase_data[i].soi_backward.yfb[6]);
                
                // partial derivative of match point with respect to moving backward leg forward in time
                Eigen::Matrix<double, 6, 1> dmp_dtau_back = stmdv3 * (stmsoib * xbp1 + pdmp_dtau_soib) + pdmp_dtau_dv3;
        
                // derivatives with respect to epoch time			
                Eigen::Matrix<double, 6, 1> dmp_dtau_soif = -(stmdv1 * (pdmp_dtau_soif + stmsoif * xbp0) + pdmp_dtau_dv1) + dmp_dtau_back;
                Eigen::Matrix<double, 6, 1> dmp_dtau_dv1 = -(pdmp_dtau_dv1 + stmdv1 * dmp_dt_soif) + dmp_dtau_back;
                Eigen::Matrix<double, 6, 1> dmp_dtau_dv2 = -dmp_dt_dv1 + dmp_dtau_back;
                Eigen::Matrix<double, 6, 1> dmp_dtau_dv3 = stmdv3 * (stmsoib * xbp1 + pdmp_dtau_soib) + pdmp_dtau_dv3 - dmp_dt_dv3;
                Eigen::Matrix<double, 6, 1> dmp_dtau_soib = stmdv3 * (stmsoib * xbp1 + pdmp_dtau_soib - dmp_dt_soib);
                
                for (int j = 0; j < col_index_dt - index_t0; j++)
                {
                    mgrad.block(row_index_mp, col_index_dt - 1 - j, 6, 1) = dmp_dtau_soif;
                }

                mgrad.block(row_index_mp, col_index_dt + 0, 6, 1) = dmp_dtau_dv1;
                mgrad.block(row_index_mp, col_index_dt + 1, 6, 1) = dmp_dtau_dv2;
                mgrad.block(row_index_mp, col_index_dt + 2, 6, 1) = dmp_dtau_dv3;
                mgrad.block(row_index_mp, col_index_dt + 3, 6, 1) = dmp_dtau_soib;
                
            
                //
                // increment indices
                //
                
                col_index_dt += 4;
                col_index_dv_unit += 9;		
                row_index_mp += 6;
                row_index_dv_unit += 3;
                
                // match point derivatives with respect to initial state and time within sphere of influence
                // if body is home body include lunar match points
                if (data->sequence[i + 1] == data->index_home_body)
                {
                    // match point derivative between soi and moon with respect to soi time
                    Eigen::Matrix<double, 6, 1> dmp_soi_moon_dtau_moon = (
                        Eigen::Map<Eigen::Matrix<double, 6, 1>>(&phase_data[i].moon_backward.yfb[42]) -
                        Eigen::Map<Eigen::Matrix<double, 6, 1>>(&phase_data[i].soi_backward.yff[42])) / data->t_scale[i + 1];

                    for (int j = 0; j < col_index_dt - index_t0; j++)
                    {
                        mgrad.block(row_index_mp, col_index_dt - 1 - j, 6, 1) = dmp_soi_moon_dtau_moon;
                    }
                    
                    // match point derivative between soi and moon with respect to moon time
                    mgrad.block(row_index_mp, col_index_dt, 6, 1) = (
                        Eigen::Map<Eigen::Matrix<double, 6, 1>>(&phase_data[i].moon_backward.yfb[42]) - 0.5 * (
                        Eigen::Map<Eigen::Matrix<double, 6, 1>>(&phase_data[i].soi_backward.ypf[0]) + 
                        Eigen::Map<Eigen::Matrix<double, 6, 1>>(&phase_data[i].moon_backward.ypb[0]))) / data->t_scale[i + 1];
                    col_index_dt++;
                    
                    // match point derivative between soi and moon with respect to initial state at soi
                    mgrad.block<6, 5>(row_index_mp, col_index_mp) = -spherical_stm(phase_data[i].soi_backward, 1, data->d_scale[i + 1], data->v_scale[i + 1]).block<6, 5>(0, 1);
                    col_index_mp += 5;

                    // lunar distance constraint
                    double ra = phase_data[i].moon_backward.xi[1];
                    double dec = phase_data[i].moon_backward.xi[2];

                    Eigen::Vector3d rm = data->ephemeris->get_position(data->index_home_moon, phase_data[i].moon_backward.ti) - data->ephemeris->get_position(data->index_home_body, phase_data[i].moon_backward.ti);
                    Eigen::Vector3d vm = data->ephemeris->get_velocity(data->index_home_moon, phase_data[i].moon_backward.ti) - data->ephemeris->get_velocity(data->index_home_body, phase_data[i].moon_backward.ti);

                    mgrad(row_index_moon, col_index_mp) = 2 * data->moon_mean_distance * cos(dec) * (rm(0) * sin(ra) - rm(2) * cos(ra));
                    mgrad(row_index_moon, col_index_mp + 1) = 2 * data->moon_mean_distance * (rm(0) * cos(ra) * sin(dec) - rm(1) * cos(dec) + rm(2) * sin(ra) * sin(dec));;
                    mgrad(row_index_moon, index_sl) = -1;
                    index_sl++;
                    
                    // match point derivatives between soi and moon and moon and body with respect to initial state at moon		
                    mgrad.block<6, 5>(row_index_mp, col_index_mp) = spherical_stm(phase_data[i].moon_backward, -1, data->d_scale[i + 1], data->v_scale[i + 1]).block<6, 5>(0, 1);
                    row_index_mp += 6;
                    
                    mgrad.block<6, 5>(row_index_mp, col_index_mp) = -spherical_stm(phase_data[i].moon_backward, 1, data->d_scale[i + 1], data->v_scale[i + 1]).block<6, 5>(0, 1);
                    col_index_mp += 5;
                    
                    // match point derivative between moon and body with respect to moon time and lunar distance constraint with respect to moon time
                    Eigen::Matrix<double, 6, 1> dmp_moon_body_dtau_moon = (
                        Eigen::Map<Eigen::Matrix<double, 6, 1>>(&phase_data[i].body_backward.yfb[42]) - 
                        Eigen::Map<Eigen::Matrix<double, 6, 1>>(&phase_data[i].moon_backward.yff[42])) / data->t_scale[i + 1];
                        
                    for (int j = 0; j < col_index_dt - index_t0; j++)
                    {
                        mgrad.block(row_index_mp, col_index_dt - 1 - j, 6, 1) = dmp_moon_body_dtau_moon;
                        mgrad(row_index_moon, col_index_dt - 1 - j) = vm(0) * 2 * (rm(0) - phase_data[i].moon_backward.yi[0]) + vm(1) * 2 * (rm(1) - phase_data[i].moon_backward.yi[1]) + vm(2) * 2 * (rm(2) - phase_data[i].moon_backward.yi[2]);
                    }

                    row_index_moon++;

                    // match point derivative between moon and body with respect to body time
                    mgrad.block(row_index_mp, col_index_dt, 6, 1) = (
                        Eigen::Map<Eigen::Matrix<double, 6, 1>>(&phase_data[i].body_backward.yfb[42]) - 0.5 * (
                        Eigen::Map<Eigen::Matrix<double, 6, 1>>(&phase_data[i].moon_backward.ypf[0]) +
                        Eigen::Map<Eigen::Matrix<double, 6, 1>>(&phase_data[i].body_backward.ypb[0]))) / data->t_scale[i + 1];                     					            
                }
                else
                {
                    // match point derivative between soi and body with respect to initial state at soi
                    mgrad.block<6, 5>(row_index_mp, col_index_mp) = -spherical_stm(phase_data[i].soi_backward, 1, data->d_scale[i + 1], data->v_scale[i + 1]).block<6, 5>(0, 1);
                    
                    // match point derivative between soi and body with respect to soi time
                    Eigen::Matrix<double, 6, 1> dmp_soi_body_dtau_soi = (
                        Eigen::Map<Eigen::Matrix<double, 6, 1>>(&phase_data[i].body_backward.yfb[42]) -
                        Eigen::Map<Eigen::Matrix<double, 6, 1>>(&phase_data[i].soi_backward.yff[42])) / data->t_scale[i + 1];
                        
                    for (int j = 0; j < col_index_dt - index_t0; j++)
                    {
                        mgrad.block(row_index_mp, col_index_dt - 1 - j, 6, 1) = dmp_soi_body_dtau_soi;
                    }
                    
                    // match point derivative between soi and body with respect to body time
                    mgrad.block(row_index_mp, col_index_dt, 6, 1) = (
                        Eigen::Map<Eigen::Matrix<double, 6, 1>>(&phase_data[i].body_backward.yfb[42]) - 0.5 * (
                        Eigen::Map<Eigen::Matrix<double, 6, 1>>(&phase_data[i].soi_backward.ypf[0]) + 
                        Eigen::Map<Eigen::Matrix<double, 6, 1>>(&phase_data[i].body_backward.ypb[0]))) / data->t_scale[i + 1];							
                    col_index_mp += 5;
                }
                
                // match point derivatives between moon or soi and body with respect to initial state at body
                // if last phase radius is not a decision variable and periapsis derivatives are calculated	
                if (i == num_phases - 1)
                {
                    mgrad.block<6, 5>(row_index_mp, col_index_mp) = spherical_stm(phase_data[i].body_backward, -1, data->d_scale[i + 1], data->v_scale[i + 1]).block<6, 5>(0, 1);
                    mgrad.block<1, 5>(row_index_periapsis, col_index_mp) = spherical_derivative_r_dot_v(phase_data[i].body_backward.xi).block<5, 1>(1, 0);
                    row_index_periapsis++;
                    col_index_mp += 6;
                }
                else
                {
                    mgrad.block<6, 6>(row_index_mp, col_index_mp) = spherical_stm(phase_data[i].body_backward, -1, data->d_scale[i + 1], data->v_scale[i + 1]);
                    row_index_mp += 6;
                }
                
                col_index_dt++;	
            }

            // derivatives for total time
            for (int i = 0; i < num_dt; i++)
            {
                mgrad(row_index_tof, index_t0 + 1 + i) = 1;
            }

            // derivatives for slack variables    
            mgrad(row_index_inc + 0, index_sl + 0) = 1;
            mgrad(row_index_inc + 1, index_sl + 1) = -1;
            mgrad(row_index_inc + 2, index_sl + 2) = 1;
            mgrad(row_index_inc + 3, index_sl + 3) = -1;
            mgrad(row_index_tof, index_sl + 4) = 1;

            if (data->min_time > 0)
            {
                mgrad(row_index_min_time, index_sl + 5) = -1;

                // derivatives for total time
                for (int i = 0; i < num_dt; i++)
                {
                    mgrad(row_index_min_time, index_t0 + 1 + i) = 1;
                }
                if (data->max_c3 > 0)
                {
                    mgrad(row_index_c3, index_sl + 6) = 1;
                }
            }
            else
            {
                if (data->max_c3 > 0)
                {
                    mgrad(row_index_c3, index_sl + 5) = 1;
                }
            }


            //
            //derivatives for inclination
            //
            
            // ra, dec, vrho, vra, vdec
            //  0    1     2    3     4
            // -5   -4    -3   -2    -1

            // launch

            mgrad(row_index_inc + 0, 0) = -cos(x[1]) * cos(x[4]) * cos(x[3] - x[0]);
            mgrad(row_index_inc + 1, 0) = -cos(x[1]) * cos(x[4]) * cos(x[3] - x[0]);

            mgrad(row_index_inc + 0, 1) = -sin(x[1]) * cos(x[4]) * sin(x[3] - x[0]);
            mgrad(row_index_inc + 1, 1) = -sin(x[1]) * cos(x[4]) * sin(x[3] - x[0]);

            mgrad(row_index_inc + 0, 3) = cos(x[1]) * cos(x[4]) * cos(x[0] - x[3]);
            mgrad(row_index_inc + 1, 3) = cos(x[1]) * cos(x[4]) * cos(x[0] - x[3]);

            mgrad(row_index_inc + 0, 4) = -cos(x[1]) * sin(x[4]) * sin(x[3] - x[0]);
            mgrad(row_index_inc + 1, 4) = -cos(x[1]) * sin(x[4]) * sin(x[3] - x[0]);

            // arrival

            mgrad(row_index_inc + 2, index_t0 - 5) = -cos(x[index_t0 - 4]) * cos(x[index_t0 - 1]) * cos(x[index_t0 - 2] - x[index_t0 - 5]);
            mgrad(row_index_inc + 3, index_t0 - 5) = -cos(x[index_t0 - 4]) * cos(x[index_t0 - 1]) * cos(x[index_t0 - 2] - x[index_t0 - 5]);

            mgrad(row_index_inc + 2, index_t0 - 4) = -sin(x[index_t0 - 4]) * cos(x[index_t0 - 1]) * sin(x[index_t0 - 2] - x[index_t0 - 5]);
            mgrad(row_index_inc + 3, index_t0 - 4) = -sin(x[index_t0 - 4]) * cos(x[index_t0 - 1]) * sin(x[index_t0 - 2] - x[index_t0 - 5]);

            mgrad(row_index_inc + 2, index_t0 - 2) = cos(x[index_t0 - 4]) * cos(x[index_t0 - 1]) * cos(x[index_t0 - 5] - x[index_t0 - 2]);
            mgrad(row_index_inc + 3, index_t0 - 2) = cos(x[index_t0 - 4]) * cos(x[index_t0 - 1]) * cos(x[index_t0 - 5] - x[index_t0 - 2]);

            mgrad(row_index_inc + 2, index_t0 - 1) = -cos(x[index_t0 - 4]) * sin(x[index_t0 - 1]) * sin(x[index_t0 - 2] - x[index_t0 - 5]);
            mgrad(row_index_inc + 3, index_t0 - 1) = -cos(x[index_t0 - 4]) * sin(x[index_t0 - 1]) * sin(x[index_t0 - 2] - x[index_t0 - 5]);

            for (int j = 0; j < m; j++)
            {
                for (int i = 0; i < n; i++)
                {
                    grad[j * n + i] = mgrad(j, i);
                }
            }
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "error in nbody constraint function:" << '\n';
        std::cerr << e.what() << '\n';
        throw e;
    }
}

// calculate objective function of numerical model for nlopt
double FlightPlan::objective_nbody_model(unsigned n, const double* x, double* grad, void* f_data)
{
    try
    {
        FlightPlanData* data = reinterpret_cast<FlightPlanData*>(f_data);

        // initialize data
        int num_phases = data->num_bodies - 1;
        int index_t0 = 16 * num_phases + 5 * data->num_moon_flybys + 4;
        int num_dt = 6 * num_phases + data->num_moon_flybys;
        int num_dv = 3 * num_phases;

        int index_dt = index_t0 + 1;
        int index_dv = index_dt + num_dt;
        int index_dvn = index_dv + 3 * num_dv;
        int index_sl = index_dvn + num_dv;

        // calculate launch and arrival delta velocity
        double orbital_radius_launch = data->radius_min[0];
        double orbital_radius_arrival = data->radius_min[data->num_bodies - 1];

        double mu_body_launch = data->ephemeris->get_mu(data->sequence[0]);
        double mu_body_arrival = data->ephemeris->get_mu(data->sequence[data->num_bodies - 1]);

        double orbital_speed_arrival = sqrt((1.0 + data->eccentricity_arrival) / (1.0 - data->eccentricity_arrival) * mu_body_arrival / (orbital_radius_arrival / (1.0 - data->eccentricity_arrival)));

        double dvl = abs(x[2] - sqrt(mu_body_launch / orbital_radius_launch));
        double dva = abs(x[index_t0 - 3] - orbital_speed_arrival);

        double dvt = 0.0;

        if (data->max_c3 < 0)
        {
            dvt += dvl;
        }

        if (data->eccentricity_arrival >= 0)
        {
            dvt += dva;
        }

        // add dsm delta velocity
        for (int i = index_dvn; i < index_sl; i++)
        {
            dvt += x[i];
        }

        // calculate gradient
        if (grad)
        {
            for (int i = 0; i < n; i++)
            {
                grad[i] = 0.0;
            }

            if (data->max_c3 < 0)
            {
                grad[2] = 1.0;
            }

            if (data->eccentricity_arrival >= 0)
            {
                grad[index_t0 - 3] = 1.0;
            }

            for (int i = index_dvn; i < index_sl; i++)
            {
                grad[i] = 1.0;
            }
        }

        return dvt;
    }
    catch (const std::exception& e)
    {
        std::cerr << "error in nbody objective function:" << '\n';
        std::cerr << e.what() << '\n';
        throw e;
    }
}

// converts polar input state vector to cartesian state vector
std::tuple<std::vector<double>, std::vector<double>> FlightPlan::parse_input_state(const double*& x_ptr, double rho)
{
    std::vector<double> xi;
    std::vector<double> yi;
    xi.reserve(6);
    yi.reserve(6);

    if (rho == 0)
    {
        rho = *x_ptr++;
    }

    xi.push_back(rho);

    for (int j = 1; j < 6; j++)
    {
        xi.push_back(*x_ptr++);
    }

    double ra = xi[1];
    double dec = xi[2];
    double vrho = xi[3];
    double vra = xi[4];
    double vdec = xi[5];

    yi.push_back(rho * cos(ra) * cos(dec));
    yi.push_back(rho * sin(dec));
    yi.push_back(rho * sin(ra) * cos(dec));
    yi.push_back(vrho * cos(vra) * cos(vdec));
    yi.push_back(vrho * sin(vdec));
    yi.push_back(vrho * sin(vra) * cos(vdec)); 

    return { xi, yi };
}

// returns phase data for given phase
// i = phase number
// t_ptr = pointer to phase time
// x_ptr = pointer to phase state
// data = pointer to flight plan data
void FlightPlan::init_phase(PhaseData& phase_data, int i, const double*& t_ptr, const double*& x_ptr, FlightPlanData* data)
{
    // populate bodies at start and end of phase
    phase_data.b_forward = data->sequence[i];
    phase_data.b_backward = data->sequence[i + 1];
    phase_data.d_scale_forward = data->d_scale[i];
    phase_data.d_scale_backward = data->d_scale[i + 1];

    // populate body forward
    if (i == 0)
    {
        std::tie(phase_data.body_forward.xi, phase_data.body_forward.yi) = parse_input_state(x_ptr, data->radius_min[0]); 
    }
    else
    {            
        std::tie(phase_data.body_forward.xi, phase_data.body_forward.yi) = parse_input_state(x_ptr);   
    }

    phase_data.body_forward.ti = t_ptr[0];
    phase_data.body_forward.tf = 0.5 * (t_ptr[1] - t_ptr[0]) + t_ptr[0];
    t_ptr++;

    // populate moon forward
    if (data->sequence[i] == data->index_home_body)
    {
        std::tie(phase_data.moon_forward.xi, phase_data.moon_forward.yi) = parse_input_state(x_ptr, data->moon_mean_distance);

        phase_data.moon_forward.tb = 0.5 * (t_ptr[0] - t_ptr[-1]) + t_ptr[-1];
        phase_data.moon_forward.ti = t_ptr[0];
        phase_data.moon_forward.tf = 0.5 * (t_ptr[1] - t_ptr[0]) + t_ptr[0];
        t_ptr++;
    }

    // populate soi forward

    std::tie(phase_data.soi_forward.xi, phase_data.soi_forward.yi) = parse_input_state(x_ptr, data->radius_soi[i]);

    phase_data.soi_forward.tb = 0.5 * (t_ptr[0] - t_ptr[-1]) + t_ptr[-1];
    phase_data.soi_forward.ti = t_ptr[0];
    phase_data.soi_forward.tf = t_ptr[1];
    t_ptr++;

    // populate delta v
    phase_data.dv1.ti = t_ptr[0];
    phase_data.dv1.tf = t_ptr[1];

    phase_data.dv3.tb = t_ptr[1];
    phase_data.dv3.ti = t_ptr[2];

    t_ptr += 3;
    
    // populate soi backward
    std::tie(phase_data.soi_backward.xi, phase_data.soi_backward.yi) = parse_input_state(x_ptr, data->radius_soi[i + 1]);

    phase_data.soi_backward.tb = t_ptr[-1];
    phase_data.soi_backward.ti = t_ptr[0];
    phase_data.soi_backward.tf = 0.5 * (t_ptr[1] - t_ptr[0]) + t_ptr[0];
    t_ptr++;

    // populate moon backward
    if (data->sequence[i + 1] == data->index_home_body)
    {
        std::tie(phase_data.moon_backward.xi, phase_data.moon_backward.yi) = parse_input_state(x_ptr, data->moon_mean_distance);

        phase_data.moon_backward.tb = 0.5 * (t_ptr[0] - t_ptr[-1]) + t_ptr[-1];
        phase_data.moon_backward.ti = t_ptr[0];
        phase_data.moon_backward.tf = 0.5 * (t_ptr[1] - t_ptr[0]) + t_ptr[0];
        t_ptr++;
    }

    // populate body backward
    if (i == data->num_bodies - 2)
    {
        std::tie(phase_data.body_backward.xi, phase_data.body_backward.yi) = parse_input_state(x_ptr,  data->radius_min[data->num_bodies - 1]);
        x_ptr -= 5;
    }
    else
    {            
        std::tie(phase_data.body_backward.xi, phase_data.body_backward.yi) = parse_input_state(x_ptr);
        x_ptr -= 6;
    }

    phase_data.body_backward.tb = 0.5 * (t_ptr[0] - t_ptr[-1]) + t_ptr[-1];
    phase_data.body_backward.ti = t_ptr[0];
}

// adds change in velocity to state vector
std::vector<double> FlightPlan::add_dv(const double* yi, const double* dv, double dvn)
{
    std::vector<double> yf(6);
    yf[0] = yi[0];
    yf[1] = yi[1];
    yf[2] = yi[2];
    yf[3] = yi[3] + dv[0] * dvn;
    yf[4] = yi[4] + dv[1] * dvn;
    yf[5] = yi[5] + dv[2] * dvn;

    return yf;
}

// integrates trajectory
// integration_data = stores times, states, and derivatives for forward and backward direction
// forward = specifies direction of integration
// ref = specifies change in reference frame
// bref = body of desired reference frame
// eps = integration error tolerance
// grad = vector to store gradient information if it is not null
void FlightPlan::integrate(IntegrationData& data, Ephemeris* ephemeris, bool forward, double d_scale, bool ref, int bref, double eps, double* grad)
{
    int neqn;
    std::vector<double> yi = data.yi;
    std::vector<double>* yf;
    std::vector<double>* yp;
    double ti = data.ti;
    double tf;
    Equation equation;
    std::vector<double> vi;

    void(*f)(double t, double y[], double yp[], void* params);

    astrodynamics::NBodyEphemerisParams p;
    astrodynamics::NBodyRefEphemerisParams pref;

    void* params;

    if (forward)
    {
        yf = &data.yff;
        yp = &data.ypf;
        tf = data.tf;
    }
    else
    {
        yf = &data.yfb;
        yp = &data.ypb;
        tf = data.tb;
    }

    if (ref)
    {
        pref = astrodynamics::reference_frame_params(ephemeris, bref, d_scale);
        params = &pref;

        yi = astrodynamics::state_to_reference_frame(pref, yi.data());
        ti /= pref.reference_time_scale;
        tf /= pref.reference_time_scale;

        if (grad)
        {
            f = astrodynamics::n_body_df_stm_ephemeris_ref;
        }
        else
        {
            f = astrodynamics::n_body_df_ephemeris_ref;
        }       
    }
    else
    {
        p = astrodynamics::NBodyEphemerisParams({ ephemeris });
        params = &p;

        if(bref != -1)
        {
            yi = astrodynamics::state_add_body(ephemeris, yi.data(), bref, ti);
        }

        if (grad)
        {
            f = astrodynamics::n_body_df_stm_ephemeris;
        }
        else
        {
            f = astrodynamics::n_body_df_ephemeris;
        }   
    }
        
    if (grad)
    {
        neqn = 48;
        yi = astrodynamics::state_add_stm(yi.data());
    }
    else
    {
        neqn = 6;
    }
        
    equation = Equation(f, neqn, yi.data(), ti, eps, eps, params);
    equation.stepn(tf);

    yf->resize(neqn);
    equation.get_y(0, neqn, yf->data());

    if (grad)
    {
        yp->resize(6);
        equation.get_yp(0, 6, yp->data());
    }
}


// returns the state transition matrix with respect to initial state in spherical coordinates, accounts for the scaling of the reference frame
Eigen::Matrix<double, 6, 6> FlightPlan::spherical_stm(IntegrationData data, int direction, double d_scale, double v_scale)
{
    std::vector<double> x = data.xi;
    std::vector<double> yf;
    if (direction == 1)
    {
        yf = data.yff;
    }
    else
    {
        yf = data.yfb;
    }
    
    Eigen::Matrix<double, 6, 6> stm = Eigen::Map<Eigen::Matrix<double, 6, 6>>(&yf[6]);
    Eigen::Matrix<double, 6, 6> ds;

    double rho = x[0] / d_scale;
    double ra = x[1];
    double dec = x[2];
    double vrho = x[3] / v_scale;
    double vra = x[4];
    double vdec = x[5];

    ds.col(0) << cos(ra) * cos(dec) / d_scale, sin(dec) / d_scale, sin(ra) * cos(dec) / d_scale, 0.0, 0.0, 0.0;
    ds.col(1) << -rho * sin(ra) * cos(dec), 0,  rho * cos(ra) * cos(dec), 0.0, 0.0, 0.0;
    ds.col(2) << -rho * cos(ra) * sin(dec), rho * cos(dec), -rho * sin(ra) * sin(dec), 0.0, 0.0, 0.0;
    ds.col(3) << 0.0, 0.0, 0.0, cos(vra) * cos(vdec) / v_scale, sin(vdec) / v_scale, sin(vra) * cos(vdec) / v_scale;
    ds.col(4) << 0.0, 0.0, 0.0, -vrho * sin(vra) * cos(vdec), 0,  vrho * cos(vra) * cos(vdec);
    ds.col(5) << 0.0, 0.0, 0.0, -vrho * cos(vra) * sin(vdec), vrho * cos(vdec), -vrho * sin(vra) * sin(vdec);

    stm *= ds;

    return stm;
}

// returns the derivative of the dot product of r and v with respect to x in spherical coordinates
Eigen::Matrix<double, 6, 1> FlightPlan::spherical_derivative_r_dot_v(std::vector<double> x)
{
    Eigen::Matrix<double, 6, 1> ds;

    double rho = x[0];
    double ra = x[1];
    double dec = x[2];
    double vrho = x[3];
    double vra = x[4];
    double vdec = x[5];

    ds(0) = vrho * (cos(ra) * cos(dec) * cos(vra) * cos(vdec) + sin(dec) * sin(vdec) + sin(ra) * cos(dec) * sin(vra) * cos(vdec));
    ds(1) = rho * vrho * (-sin(ra) * cos(dec) * cos(vra) * cos(vdec) + cos(ra) * cos(dec) * sin(vra) * cos(vdec));
    ds(2) = rho * vrho * (cos(ra) * -sin(dec) * cos(vra) * cos(vdec) + cos(dec) * sin(vdec) + sin(ra) * -sin(dec) * sin(vra) * cos(vdec));

    ds(3) = rho * (cos(ra) * cos(dec) * cos(vra) * cos(vdec) + sin(dec) * sin(vdec) + sin(ra) * cos(dec) * sin(vra) * cos(vdec));
    ds(4) = rho * vrho * (cos(ra) * cos(dec) * -sin(vra) * cos(vdec) + sin(ra) * cos(dec) * cos(vra) * cos(vdec));
    ds(5) = rho * vrho * (cos(ra) * cos(dec) * cos(vra) * -sin(vdec) + sin(dec) * cos(vdec) + sin(ra) * cos(dec) * sin(vra) * -sin(vdec));

    return ds;
}
