#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <nlopt.hpp>

#include "Jdate.h"
#include "Equation.h"
#include "Ephemeris.h"
#include "astrodynamics.h"
#include "LunarFlightPlan.h"

LunarFlightPlan::LunarFlightPlan(Ephemeris& ephemeris)
{
	data_.ephemeris = &ephemeris;

    data_.initial_orbit = false;
    data_.free_return = false;
    data_.min_time = -1.0;
    data_.max_time = data_.ephemeris->get_max_time();
    data_.min_inclination_launch = 0.0;
    data_.max_inclination_launch = astrodynamics::pi;
    data_.n_launch = -Eigen::Vector3d::UnitY();
    data_.min_inclination_arrival = 0.0;
    data_.max_inclination_arrival = astrodynamics::pi;
    data_.n_arrival = -Eigen::Vector3d::UnitY();

    day_ = 86400.0 / data_.ephemeris->get_time_scale();
    year_ = day_ * 365.25;
}

LunarFlightPlan::~LunarFlightPlan()
{

}

//
// constraint functions
//

void LunarFlightPlan::set_mission(Jdate initial_time, bool free_return, double rp_earth, double rp_moon, double* initial_orbit)
{
    if (initial_orbit)
    {
        data_.initial_orbit = true;
        data_.initial_position = { initial_orbit[0], initial_orbit[1], initial_orbit[2] };
        data_.initial_position /= data_.ephemeris->get_distance_scale();
        data_.initial_velocity = { initial_orbit[3], initial_orbit[4], initial_orbit[5] };
        data_.initial_velocity /= data_.ephemeris->get_velocity_scale();
    }
    data_.free_return = free_return;
    data_.initial_time = data_.ephemeris->get_ephemeris_time(initial_time);
    data_.rp_earth = rp_earth / data_.ephemeris->get_distance_scale();
    data_.rp_moon = rp_moon / data_.ephemeris->get_distance_scale();
}

void LunarFlightPlan::add_min_flight_time_constraint(double min_time)
{
    data_.min_time = min_time * day_;
}

void LunarFlightPlan::add_max_flight_time_constraint(double max_time)
{
    data_.max_time = max_time * day_;
}

void LunarFlightPlan::add_inclination_constraint(bool launch, double min, double max, Eigen::Vector3d n)
{
    if (launch)
    {
        data_.min_inclination_launch = astrodynamics::pi / 180.0 * min;
        data_.max_inclination_launch = astrodynamics::pi / 180.0 * max;
        data_.n_launch = n;
    }
    else
    {
        data_.min_inclination_arrival = astrodynamics::pi / 180.0 * min;
        data_.max_inclination_arrival = astrodynamics::pi / 180.0 * max;
        data_.n_arrival = n;
    }
}

void LunarFlightPlan::init_model()
{
    double mu_earth = data_.ephemeris->get_muk();
    double mu_moon = data_.ephemeris->get_mu("Moon");

    auto [rm0, vm0] = data_.ephemeris->get_position_velocity("Moon", data_.initial_time);
    double dt = astrodynamics::pi * sqrt(pow(rm0.norm(), 3) / mu_earth);
    auto [rm1, vm1] = data_.ephemeris->get_position_velocity("Moon", data_.initial_time + dt);

    Eigen::Vector3d n = rm1.cross(vm1).normalized();

    Eigen::Vector3d r0 = -data_.rp_earth * rm1.normalized();
    Eigen::Vector3d v0 = sqrt(mu_earth * (2.0 / data_.rp_earth - 1.0 / rm1.norm())) * rm1.cross(n).normalized();

    Eigen::Vector3d r1 = (data_.rp_moon + rm1.norm()) * rm1.normalized();
    Eigen::Vector3d v1 = sqrt(mu_moon * (2.0 / data_.rp_moon - 1.0 / rm1.norm())) * rm1.cross(n).normalized() + vm1;

    auto push_back_state = [](std::vector<double>& x, Eigen::Vector3d r, Eigen::Vector3d v, bool initial_orbit, bool radius)
    {
        if (!initial_orbit)
        {
            if (radius)
            {
                x.push_back(r.norm());
            }

            x.push_back(atan2(r(2), r(0)));
            x.push_back(asin(r(1) / r.norm()));
        }

        x.push_back(v.norm());
        x.push_back(atan2(v(2), v(0)));
        x.push_back(asin(v(1) / v.norm()));
    };

    auto calc_orbit_delta_time = [](Eigen::Vector3d rm, Eigen::Vector3d r, Eigen::Vector3d v, double mu, double time_scale)
    {
        double two_minutes = 120.0 / time_scale;
        double dt = 0.0;
        std::vector<double> d;
        for (int i = 0; i < 200; i++)
        {
            std::tie(r, v) = astrodynamics::kepler(r, v, two_minutes, mu, 1e-8);
            d.push_back(r.dot(rm) / (r.norm() * rm.norm()));
            dt += two_minutes;
            if (d.size() > 2)
            {
                if (d[i] > d[i - 1] && d[i - 2] > d[i - 1])
                {
                    break;
                }
            }
        }
        std::tie(r, v) = astrodynamics::kepler(r, v, -two_minutes, mu, 1e-8);

        return std::tuple(dt - two_minutes, v);
    };

    if (data_.free_return)
    {
        double orbit_delta_time;
        Eigen::Vector3d initial_velocity;

        if (data_.initial_orbit)
        {
            std::tie(orbit_delta_time, initial_velocity) = calc_orbit_delta_time(rm1, data_.initial_position, data_.initial_velocity, mu_earth, data_.ephemeris->get_time_scale());
            initial_velocity = sqrt(mu_earth * (2.0 / data_.rp_earth - 1.0 / rm1.norm())) * initial_velocity.normalized();
            push_back_state(x_, r0, initial_velocity, true, false);
        }
        else
        {
            push_back_state(x_, r0, v0, false, false);
        }
    
        push_back_state(x_, r1, v1, false, true);
        push_back_state(x_, r0, v0, false, false);

        if (data_.initial_orbit)
        {
            x_.push_back(orbit_delta_time);
        }
        else
        {
            x_.push_back(data_.initial_time);
        }

        x_.push_back(dt);
        x_.push_back(dt);
        x_.push_back(0.0);
        x_.push_back(0.0);
        x_.push_back(0.0);
        x_.push_back(0.0);

        if (!data_.initial_orbit)
        {
            x_.push_back(0.0);
            x_.push_back(0.0);
        }
    }
    else
    {
        double orbit_delta_time;
        Eigen::Vector3d initial_velocity;

        if (data_.initial_orbit)
        {
            std::tie(orbit_delta_time, initial_velocity) = calc_orbit_delta_time(rm1, data_.initial_position, data_.initial_velocity, mu_earth, data_.ephemeris->get_time_scale());
            initial_velocity = sqrt(mu_earth * (2.0 / data_.rp_earth - 1.0 / rm1.norm())) * initial_velocity.normalized();
            push_back_state(x_, r0, initial_velocity, true, false);
        }
        else
        {
            push_back_state(x_, r0, v0, false, false);
        }

        push_back_state(x_, r1, v1, false, false);

        if (data_.initial_orbit)
        {
            x_.push_back(orbit_delta_time);
        }
        else
        {
            x_.push_back(data_.initial_time);
        }

        x_.push_back(dt);
        x_.push_back(0.0);
        x_.push_back(0.0);
        x_.push_back(0.0);
        x_.push_back(0.0);

        if (!data_.initial_orbit)
        {
            x_.push_back(0.0);
            x_.push_back(0.0);
        }
    }
}

void LunarFlightPlan::run_model(int max_eval, double eps, double eps_t, double eps_f)
{
    // calculate number of constraints and decision variables
    int n = 18;
    int m = 12;

    if (data_.initial_orbit)
    {
        n -= 4;
        m -= 2;
    }

    if (data_.free_return)
    {
        n += 7;
        m += 6;
    }

    // initialize optimizer
    data_.eps = eps;

    std::vector<double> tol(m, eps_t);

    init_model();
    auto [lower_bounds, upper_bounds] = bounds();

    double minf;
    opt_ = nlopt::opt("LD_SLSQP", n);
    opt_.set_lower_bounds(lower_bounds);
    opt_.set_upper_bounds(upper_bounds);
    opt_.set_min_objective(objective, &data_);
    opt_.add_equality_mconstraint(constraints, &data_, tol);
    opt_.set_maxeval(max_eval);
    opt_.set_ftol_abs(eps_f);
    opt_.optimize(x_, minf);
}

LunarFlightPlan::Result LunarFlightPlan::output_result(double eps)
{
    return Result();
}

std::tuple<std::vector<double>, std::vector<double>> LunarFlightPlan::bounds()
{
    std::vector<double> lower_bounds;
    std::vector<double> upper_bounds;

    return { lower_bounds, upper_bounds };
}

void LunarFlightPlan::constraints(unsigned m, double* result, unsigned n, const double* x, double* grad, void* f_data)
{

}

double LunarFlightPlan::objective(unsigned n, const double* x, double* grad, void* f_data)
{
    return 0.0;
}
