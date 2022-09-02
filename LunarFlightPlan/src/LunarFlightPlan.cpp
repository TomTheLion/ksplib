#include <iostream>
#include <iomanip>
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

    data_.min_time = 0.0;
    data_.max_time = data_.ephemeris->get_max_time();
    data_.min_inclination_launch = 0.0;
    data_.max_inclination_launch = astrodynamics::pi;
    data_.n_launch = -Eigen::Vector3d::UnitY();
    data_.min_inclination_arrival = 0.0;
    data_.max_inclination_arrival = astrodynamics::pi;
    data_.n_arrival = -Eigen::Vector3d::UnitY();
    data_.sun = "Sun";
    data_.moon = "Moon";
    data_.moon_radius = 1731100.0 / data_.ephemeris->get_distance_scale();
    data_.moon_soi_radius = 50000000.0 / data_.ephemeris->get_distance_scale();

    day_ = 86400.0 / data_.ephemeris->get_time_scale();
}

LunarFlightPlan::~LunarFlightPlan()
{

}

//
// constraint functions
//

void LunarFlightPlan::set_mission(Jdate initial_time, TrajectoryMode mode, double rp_earth, double rp_moon, double e_moon)
{
    data_.mode = mode;
    data_.initial_time = data_.ephemeris->get_ephemeris_time(initial_time);
    data_.rp_earth = rp_earth / data_.ephemeris->get_distance_scale();
    data_.rp_moon = rp_moon / data_.ephemeris->get_distance_scale();
    e_moon_ = e_moon;
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
    double mu_moon = data_.ephemeris->get_mu(data_.moon);

    auto [rm, vm] = data_.ephemeris->get_position_velocity(data_.moon, data_.initial_time);
    double dt = sqrt(0.5 * pow(rm.norm(), 3) / mu_earth);
    if (data_.mode != TrajectoryMode::RETURN)
    {
        std::tie(rm, vm) = data_.ephemeris->get_position_velocity(data_.moon, data_.initial_time + dt);
    }

    Eigen::Vector3d n = rm.cross(vm).normalized();

    double min_e_earth = (rm.norm() - data_.rp_earth) / (rm.norm() + data_.rp_earth);

    Eigen::Vector3d r0 = -data_.rp_earth * rm.normalized();
    Eigen::Vector3d v0 = sqrt((1.0 + min_e_earth) * mu_earth / data_.rp_earth) * rm.cross(n).normalized();

    Eigen::Vector3d r1, v1;

    if (data_.mode == TrajectoryMode::RETURN && data_.n_launch.dot(n) > 0)
    {
        v1 = -sqrt((1.0 + e_moon_) * mu_moon / data_.rp_moon) * rm;
        r1 = data_.rp_moon * v1.cross(n).normalized();
    }
    else
    {
        r1 = data_.rp_moon * rm.normalized();
        v1 = sqrt((1.0 + e_moon_) * mu_moon / data_.rp_moon) * rm.cross(n).normalized();
    }

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

    if (data_.mode == TrajectoryMode::FREE_RETURN)
    {
        push_back_state(x_, r0, v0, false);
        push_back_state(x_, r1, v1, true);
        push_back_state(x_, r0, v0, false);

        x_.push_back(dt);
        x_.push_back(dt);
        x_.push_back(0.0);
        x_.push_back(0.0);
        x_.push_back(0.0);
        x_.push_back(0.0);
        x_.push_back(0.0);
        x_.push_back(0.0);
    }
    else
    {
        push_back_state(x_, r0, v0, false);
        push_back_state(x_, r1, v1, false);

        x_.push_back(dt);
        x_.push_back(0.0);
        x_.push_back(0.0);
        x_.push_back(0.0);
        x_.push_back(0.0);
        x_.push_back(0.0);
        x_.push_back(0.0);
    }
}

void LunarFlightPlan::run_model(int max_eval, double eps, double eps_t, double eps_x)
{
    // calculate number of constraints and decision variables
    int n = 17;
    int m = 14;

    if (data_.mode == TrajectoryMode::FREE_RETURN)
    {
        n += 7;
        m += 7;
    }

    // initialize optimizer
    data_.eps = eps;

    std::vector<double> tol(m, eps_t);

    auto [lower_bounds, upper_bounds] = bounds();

    double minf;
    opt_ = nlopt::opt("LD_SLSQP", n);
    opt_.set_lower_bounds(lower_bounds);
    opt_.set_upper_bounds(upper_bounds);

    switch (data_.mode)
    {
    case TrajectoryMode::FREE_RETURN:
        opt_.set_min_objective(free_return_objective, &data_);
        opt_.add_equality_mconstraint(free_return_constraints, &data_, tol);
        break;
    case TrajectoryMode::LEAVE:
        opt_.set_min_objective(leave_objective, &data_);
        opt_.add_equality_mconstraint(leave_constraints, &data_, tol);
        break;
    case TrajectoryMode::RETURN:
        opt_.set_min_objective(return_objective, &data_);
        opt_.add_equality_mconstraint(return_constraints, &data_, tol);
        break;
    }
    
    opt_.set_maxeval(max_eval);
    opt_.set_xtol_abs(eps_x);
    opt_.optimize(x_, minf);
}

LunarFlightPlan::Result LunarFlightPlan::output_result(double eps)
{
    Result result;

    int n = 17;
    int m = 14;

    if (data_.mode == TrajectoryMode::FREE_RETURN)
    {
        n += 7;
        m += 7;
    }

    data_.eps = eps;

    result.nlopt_constraints.resize(m);
    result.r.resize(3);
    result.v.resize(3);
    result.rsun.resize(3);
    result.vsun.resize(3);
    result.rmoon.resize(3);
    result.vmoon.resize(3);

    try
    {
        result.nlopt_code = opt_.last_optimize_result();
        result.nlopt_value = opt_.last_optimum_value();
        result.nlopt_num_evals = opt_.get_numevals();
    }
    catch (const std::exception& e)
    {

    }

    result.nlopt_solution = x_;

    result.time_scale = data_.ephemeris->get_time_scale();
    result.distance_scale = data_.ephemeris->get_distance_scale();
    result.velocity_scale = data_.ephemeris->get_velocity_scale();

    result.bodies = data_.ephemeris->get_bodies();
    result.muk = data_.ephemeris->get_muk();
    result.mu = data_.ephemeris->get_mu();

    const double* x_ptr = x_.data();
    astrodynamics::NBodyEphemerisParams p({ data_.ephemeris });

    auto output = [this, &result](Equation& equation, int leg, double ti, double tf)
    {
        int steps = 500;
        std::vector<double> yf(6);

        for (int i = 0; i <= steps; i++)
        {
            double dt = (tf - ti) * static_cast<double>(i) / steps;
            double t = ti + dt;
            Jdate jd = data_.ephemeris->get_jdate(t);
            result.julian_time.push_back(jd.get_julian_date());
            result.kerbal_time.push_back(jd.get_kerbal_time());
            result.leg.push_back(leg);

            equation.step(t);
            equation.get_y(0, 6, yf.data());

            auto [rs, vs] = data_.ephemeris->get_position_velocity(data_.sun, t);
            auto [rm, vm] = data_.ephemeris->get_position_velocity(data_.moon, t);

            result.r[0].push_back(yf[0]);
            result.r[1].push_back(yf[1]);
            result.r[2].push_back(yf[2]);
            result.v[0].push_back(yf[3]);
            result.v[1].push_back(yf[4]);
            result.v[2].push_back(yf[5]);

            result.rsun[0].push_back(rs(0));
            result.rsun[1].push_back(rs(1));
            result.rsun[2].push_back(rs(2));
            result.vsun[0].push_back(vs(0));
            result.vsun[1].push_back(vs(1));
            result.vsun[2].push_back(vs(2));

            result.rmoon[0].push_back(rm(0));
            result.rmoon[1].push_back(rm(1));
            result.rmoon[2].push_back(rm(2));
            result.vmoon[0].push_back(vm(0));
            result.vmoon[1].push_back(vm(1));
            result.vmoon[2].push_back(vm(2));
        }
    };

    switch (data_.mode)
    {
    case TrajectoryMode::FREE_RETURN:
        {
            free_return_constraints(m, result.nlopt_constraints.data(), n, x_.data(), NULL, &data_);

            // parse inputs
            std::vector<double> y0 = cartesian_state(x_ptr, data_.rp_earth);
            std::vector<double> y1 = cartesian_state(x_ptr);
            std::vector<double> y2 = cartesian_state(x_ptr, data_.rp_earth);

            double t0 = data_.initial_time;
            double dt1 = *x_ptr++;
            double dt2 = *x_ptr++;

            auto [rm, vm] = data_.ephemeris->get_position_velocity(data_.moon, t0 + dt1);

            y1[0] += rm(0);
            y1[1] += rm(1);
            y1[2] += rm(2);
            y1[3] += vm(0);
            y1[4] += vm(1);
            y1[5] += vm(2);

            // integrate
            Equation equation0 = Equation(astrodynamics::n_body_df_ephemeris, y0.size(), y0.data(), t0, data_.eps, data_.eps, &p);
            Equation equation1 = Equation(astrodynamics::n_body_df_ephemeris, y1.size(), y1.data(), t0 + dt1, data_.eps, data_.eps, &p);
            Equation equation2 = Equation(astrodynamics::n_body_df_ephemeris, y2.size(), y2.data(), t0 + dt1 + dt2, data_.eps, data_.eps, &p);

            output(equation0, 0, t0, t0 + 0.5 * dt1);
            output(equation1, 1, t0 + 0.5 * dt1, t0 + dt1);
            output(equation1, 2, t0 + dt1, t0 + dt1 + 0.5 * dt2);
            output(equation2, 3, t0 + dt1 + 0.5 * dt2, t0 + dt1 + dt2);
            break;
        }
    case TrajectoryMode::LEAVE:
        {
            leave_constraints(m, result.nlopt_constraints.data(), n, x_.data(), NULL, &data_);

            // parse inputs
            std::vector<double> y0 = cartesian_state(x_ptr, data_.rp_earth);
            std::vector<double> y1 = cartesian_state(x_ptr, data_.rp_moon);

            double t0 = data_.initial_time;
            double dt = *x_ptr++;

            auto [rm, vm] = data_.ephemeris->get_position_velocity(data_.moon, t0 + dt);

            y1[0] += rm(0);
            y1[1] += rm(1);
            y1[2] += rm(2);
            y1[3] += vm(0);
            y1[4] += vm(1);
            y1[5] += vm(2);

            // integrate
            Equation equation0 = Equation(astrodynamics::n_body_df_ephemeris, y0.size(), y0.data(), t0, data_.eps, data_.eps, &p);
            Equation equation1 = Equation(astrodynamics::n_body_df_ephemeris, y1.size(), y1.data(), t0 + dt, data_.eps, data_.eps, &p);

            output(equation0, 0, t0, t0 + 0.5 * dt);
            output(equation1, 1, t0 + 0.5 * dt, t0 + dt);
            break;
        }
    case TrajectoryMode::RETURN:
        {
            return_constraints(m, result.nlopt_constraints.data(), n, x_.data(), NULL, &data_);

            // parse inputs
            std::vector<double> y1 = cartesian_state(x_ptr, data_.rp_earth);
            std::vector<double> y0 = cartesian_state(x_ptr, data_.rp_moon);

            double t0 = data_.initial_time;
            double dt = *x_ptr++;

            auto [rm, vm] = data_.ephemeris->get_position_velocity(data_.moon, t0);

            y0[0] += rm(0);
            y0[1] += rm(1);
            y0[2] += rm(2);
            y0[3] += vm(0);
            y0[4] += vm(1);
            y0[5] += vm(2);

            // integrate
            Equation equation0 = Equation(astrodynamics::n_body_df_ephemeris, y0.size(), y0.data(), t0, data_.eps, data_.eps, &p);
            Equation equation1 = Equation(astrodynamics::n_body_df_ephemeris, y1.size(), y1.data(), t0 + dt, data_.eps, data_.eps, &p);

            output(equation0, 0, t0, t0 + 0.5 * dt);
            output(equation1, 1, t0 + 0.5 * dt, t0 + dt);
            break;
        }
    }

    return result;
}

std::tuple<std::vector<double>, std::vector<double>> LunarFlightPlan::bounds()
{
    std::vector<double> lower_bounds;
    std::vector<double> upper_bounds;

    double mu_earth = data_.ephemeris->get_muk();
    double mu_moon = data_.ephemeris->get_mu(data_.moon);
    double moon_radius = data_.moon_radius;
    double moon_soi_radius = data_.moon_soi_radius;
    double d_scale = data_.ephemeris->get_distance_scale();
    double v_scale = data_.ephemeris->get_velocity_scale();

    double rm = data_.ephemeris->get_position(data_.moon, data_.initial_time).norm();

    double min_e_moon = (moon_soi_radius - data_.rp_moon) / (moon_soi_radius + data_.rp_moon);
    double min_e_earth = (rm - data_.rp_earth) / (rm + data_.rp_earth);
 
    auto push_back_state_bounds = [moon_radius, moon_soi_radius, d_scale, v_scale]
        (std::vector<double>& lower_bounds, std::vector<double>& upper_bounds, double mu, double rp, double e, bool radius)
    {
        if (radius)
        {
            lower_bounds.push_back(moon_radius);
            upper_bounds.push_back(0.5 * moon_soi_radius);
        }

        lower_bounds.push_back(-2.0 * astrodynamics::pi);
        lower_bounds.push_back(-2.0 * astrodynamics::pi);

        upper_bounds.push_back(2.0 * astrodynamics::pi);
        upper_bounds.push_back(2.0 * astrodynamics::pi);


        lower_bounds.push_back(0.999 * sqrt((1.0 + e) * mu / rp));
        lower_bounds.push_back(-2.0 * astrodynamics::pi);
        lower_bounds.push_back(-2.0 * astrodynamics::pi);

        upper_bounds.push_back(sqrt(3.0 * mu / rp));
        upper_bounds.push_back(2.0 * astrodynamics::pi);
        upper_bounds.push_back(2.0 * astrodynamics::pi);
    };

    if (data_.mode == TrajectoryMode::FREE_RETURN)
    {
        push_back_state_bounds(lower_bounds, upper_bounds, mu_earth, data_.rp_earth, min_e_earth, false);
        push_back_state_bounds(lower_bounds, upper_bounds, mu_moon, data_.rp_moon, min_e_moon, true);
        push_back_state_bounds(lower_bounds, upper_bounds, mu_earth, data_.rp_earth, min_e_earth, false);

        lower_bounds.push_back(day_);
        lower_bounds.push_back(day_);
        lower_bounds.push_back(0.0);
        lower_bounds.push_back(0.0);
        lower_bounds.push_back(0.0);
        lower_bounds.push_back(0.0);
        lower_bounds.push_back(0.0);
        lower_bounds.push_back(0.0);
        upper_bounds.push_back(HUGE_VAL);
        upper_bounds.push_back(HUGE_VAL);
        upper_bounds.push_back(HUGE_VAL);
        upper_bounds.push_back(HUGE_VAL);
        upper_bounds.push_back(HUGE_VAL);
        upper_bounds.push_back(HUGE_VAL);
        upper_bounds.push_back(HUGE_VAL);
        upper_bounds.push_back(HUGE_VAL);
    }
    else
    {
        push_back_state_bounds(lower_bounds, upper_bounds, mu_earth, data_.rp_earth, min_e_earth, false);
        push_back_state_bounds(lower_bounds, upper_bounds, mu_moon, data_.rp_moon, min_e_moon, false);

        lower_bounds.push_back(day_);
        lower_bounds.push_back(0.0);
        lower_bounds.push_back(0.0);
        lower_bounds.push_back(0.0);
        lower_bounds.push_back(0.0);
        lower_bounds.push_back(0.0);
        lower_bounds.push_back(0.0);
        upper_bounds.push_back(HUGE_VAL);
        upper_bounds.push_back(HUGE_VAL);
        upper_bounds.push_back(HUGE_VAL);
        upper_bounds.push_back(HUGE_VAL);
        upper_bounds.push_back(HUGE_VAL);
        upper_bounds.push_back(HUGE_VAL);
        upper_bounds.push_back(HUGE_VAL);
    }

    return { lower_bounds, upper_bounds };
}

std::vector<double> LunarFlightPlan::cartesian_state(const double*& x_ptr, double rho)
{
    std::vector<double> y;
    y.reserve(6);

    if (rho == 0)
    {
        rho = *x_ptr++;
    }

    double ra = *x_ptr++;
    double dec = *x_ptr++;
    double vrho = *x_ptr++;
    double vra = *x_ptr++;
    double vdec = *x_ptr++;

    y.push_back(rho * cos(ra) * cos(dec));
    y.push_back(rho * sin(dec));
    y.push_back(rho * sin(ra) * cos(dec));
    y.push_back(vrho * cos(vra) * cos(vdec));
    y.push_back(vrho * sin(vdec));
    y.push_back(vrho * sin(vra) * cos(vdec));

    return y;
}

void LunarFlightPlan::free_return_constraints(unsigned m, double* result, unsigned n, const double* x, double* grad, void* f_data)
{
    try
    {
        FlightPlanData* data = reinterpret_cast<FlightPlanData*>(f_data);

        const double* x_ptr = x;
        double* result_ptr = result;

        // parse inputs
        std::vector<double> y0 = cartesian_state(x_ptr, data->rp_earth);
        std::vector<double> y1 = cartesian_state(x_ptr);
        std::vector<double> y2 = cartesian_state(x_ptr, data->rp_earth);

        double t0 = data->initial_time;
        double dt1 = *x_ptr++;
        double dt2 = *x_ptr++;

        auto [rm, vm] = data->ephemeris->get_position_velocity(data->moon, t0 + dt1);

        y1[0] += rm(0);
        y1[1] += rm(1);
        y1[2] += rm(2);
        y1[3] += vm(0);
        y1[4] += vm(1);
        y1[5] += vm(2);

        double min_launch_inc_slack = *x_ptr++;
        double max_launch_inc_slack = *x_ptr++;
        double min_arrival_inc_slack = *x_ptr++;
        double max_arrival_inc_slack = *x_ptr++;
        double min_time_slack = *x_ptr++;
        double max_time_slack = *x_ptr++;

        // integrate
        astrodynamics::NBodyEphemerisParams p({ data->ephemeris });

        Equation equation0 = Equation(astrodynamics::n_body_df_ephemeris, y0.size(), y0.data(), t0, data->eps, data->eps, &p);
        Equation equation1 = Equation(astrodynamics::n_body_df_ephemeris, y1.size(), y1.data(), t0 + dt1, data->eps, data->eps, &p);
        Equation equation2 = Equation(astrodynamics::n_body_df_ephemeris, y2.size(), y2.data(), t0 + dt1 + dt2, data->eps, data->eps, &p);

        // calculate constraints

        // match point constraints
        std::vector<double> yff(6);
        std::vector<double> yfb(6);

        equation0.step(t0 + 0.5 * dt1);
        equation0.get_y(0, 6, yff.data());

        equation1.step(t0 + 0.5 * dt1);
        equation1.get_y(0, 6, yfb.data());

        for (int i = 0; i < 6; i++)
        {
            *result_ptr++ = yff[i] - yfb[i];
        }

        // second leg
        equation1.step(t0 + dt1 + 0.5 * dt2);
        equation1.get_y(0, 6, yff.data());

        equation2.step(t0 + dt1 + 0.5 * dt2);
        equation2.get_y(0, 6, yfb.data());

        for (int i = 0; i < 6; i++)
        {
            *result_ptr++ = yff[i] - yfb[i];
        }

        auto calc_fpa = [](double ra, double dec, double vra, double vdec)
        {
            return cos(dec) * cos(vdec) * cos(ra - vra) + sin(dec) * sin(vdec);
        };

        // flight path angle constraints
        *result_ptr++ = calc_fpa(x[0], x[1], x[3], x[4]);
        *result_ptr++ = calc_fpa(x[6], x[7], x[9], x[10]);
        *result_ptr++ = calc_fpa(x[11], x[12], x[14], x[15]);

        // inclination constraints
        auto calc_h = [](double ra, double dec, double vra, double vdec)
        {
            Eigen::Vector3d r = { cos(ra) * cos(dec), sin(dec), sin(ra) * cos(dec) };
            Eigen::Vector3d v = { cos(vra) * cos(vdec), sin(vdec), sin(vra) * cos(vdec) };
            return r.cross(v);
        };

        double cosi_launch = data->n_launch.dot(calc_h(x[0], x[1], x[3], x[4]));
        double cosi_arrival = data->n_arrival.dot(calc_h(x[11], x[12], x[14], x[15]));

        *result_ptr++ = cosi_launch - cos(data->min_inclination_launch) + min_launch_inc_slack;
        *result_ptr++ = cosi_launch - cos(data->max_inclination_launch) - max_launch_inc_slack;

        *result_ptr++ = cosi_arrival - cos(data->min_inclination_arrival) + min_arrival_inc_slack;
        *result_ptr++ = cosi_arrival - cos(data->max_inclination_arrival) - max_arrival_inc_slack;

        // time constraints
        *result_ptr++ = dt1 + dt2 - data->min_time - min_time_slack;
        *result_ptr++ = dt1 + dt2 - data->max_time + max_time_slack;

        // calculate gradients
        if (grad)
        {
            constraint_numerical_gradient(m, n, x, grad, f_data, free_return_constraints);
        }
    }   
    catch (const std::exception& e)
    {
        std::cerr << "error in constraint function:" << '\n';
        std::cerr << e.what() << '\n';
        throw e;
    }
}

void LunarFlightPlan::leave_constraints(unsigned m, double* result, unsigned n, const double* x, double* grad, void* f_data)
{
    try
    {
        FlightPlanData* data = reinterpret_cast<FlightPlanData*>(f_data);

        const double* x_ptr = x;
        double* result_ptr = result;

        // parse inputs
        std::vector<double> y0 = cartesian_state(x_ptr, data->rp_earth);
        std::vector<double> y1 = cartesian_state(x_ptr, data->rp_moon);

        double t0 = data->initial_time;
        double dt = *x_ptr++;

        auto [rm, vm] = data->ephemeris->get_position_velocity(data->moon, t0 + dt);

        y1[0] += rm(0);
        y1[1] += rm(1);
        y1[2] += rm(2);
        y1[3] += vm(0);
        y1[4] += vm(1);
        y1[5] += vm(2);

        double min_launch_inc_slack = *x_ptr++;
        double max_launch_inc_slack = *x_ptr++;
        double min_arrival_inc_slack = *x_ptr++;
        double max_arrival_inc_slack = *x_ptr++;
        double min_time_slack = *x_ptr++;
        double max_time_slack = *x_ptr++;

        // integrate
        astrodynamics::NBodyEphemerisParams p({ data->ephemeris });

        Equation equation0 = Equation(astrodynamics::n_body_df_ephemeris, y0.size(), y0.data(), t0, data->eps, data->eps, &p);
        Equation equation1 = Equation(astrodynamics::n_body_df_ephemeris, y1.size(), y1.data(), t0 + dt, data->eps, data->eps, &p);

        // calculate constraints

        // match point constraints
        std::vector<double> yff(6);
        std::vector<double> yfb(6);

        equation0.step(t0 + 0.5 * dt);
        equation0.get_y(0, 6, yff.data());

        equation1.step(t0 + 0.5 * dt);
        equation1.get_y(0, 6, yfb.data());

        for (int i = 0; i < 6; i++)
        {
            *result_ptr++ = yff[i] - yfb[i];
        }

        auto calc_fpa = [](double ra, double dec, double vra, double vdec)
        {
            return cos(dec) * cos(vdec) * cos(ra - vra) + sin(dec) * sin(vdec);
        };

        // flight path angle constraints
        *result_ptr++ = calc_fpa(x[0], x[1], x[3], x[4]);
        *result_ptr++ = calc_fpa(x[5], x[6], x[8], x[9]);

        // inclination constraints
        auto calc_h = [](double ra, double dec, double vra, double vdec)
        {
            Eigen::Vector3d r = { cos(ra) * cos(dec), sin(dec), sin(ra) * cos(dec) };
            Eigen::Vector3d v = { cos(vra) * cos(vdec), sin(vdec), sin(vra) * cos(vdec) };
            return r.cross(v);
        };

        double cosi_launch = data->n_launch.dot(calc_h(x[0], x[1], x[3], x[4]));
        double cosi_arrival = data->n_arrival.dot(calc_h(x[5], x[6], x[8], x[9]));

        *result_ptr++ = cosi_launch - cos(data->min_inclination_launch) + min_launch_inc_slack;
        *result_ptr++ = cosi_launch - cos(data->max_inclination_launch) - max_launch_inc_slack;

        *result_ptr++ = cosi_arrival - cos(data->min_inclination_arrival) + min_arrival_inc_slack;
        *result_ptr++ = cosi_arrival - cos(data->max_inclination_arrival) - max_arrival_inc_slack;

        // time constraints
        *result_ptr++ = dt - data->min_time - min_time_slack;
        *result_ptr++ = dt - data->max_time + max_time_slack;

        // calculate gradients
        if (grad)
        {
            constraint_numerical_gradient(m, n, x, grad, f_data, leave_constraints);
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "error in constraint function:" << '\n';
        std::cerr << e.what() << '\n';
        throw e;
    }
}

void LunarFlightPlan::return_constraints(unsigned m, double* result, unsigned n, const double* x, double* grad, void* f_data)
{
    try
    {
        FlightPlanData* data = reinterpret_cast<FlightPlanData*>(f_data);

        const double* x_ptr = x;
        double* result_ptr = result;

        // parse inputs
        std::vector<double> y1 = cartesian_state(x_ptr, data->rp_earth);
        std::vector<double> y0 = cartesian_state(x_ptr, data->rp_moon);

        double t0 = data->initial_time;
        double dt = *x_ptr++;

        auto [rm, vm] = data->ephemeris->get_position_velocity(data->moon, t0);

        y0[0] += rm(0);
        y0[1] += rm(1);
        y0[2] += rm(2);
        y0[3] += vm(0);
        y0[4] += vm(1);
        y0[5] += vm(2);

        double min_launch_inc_slack = *x_ptr++;
        double max_launch_inc_slack = *x_ptr++;
        double min_arrival_inc_slack = *x_ptr++;
        double max_arrival_inc_slack = *x_ptr++;
        double min_time_slack = *x_ptr++;
        double max_time_slack = *x_ptr++;

        // integrate
        astrodynamics::NBodyEphemerisParams p({ data->ephemeris });

        Equation equation0 = Equation(astrodynamics::n_body_df_ephemeris, y0.size(), y0.data(), t0, data->eps, data->eps, &p);
        Equation equation1 = Equation(astrodynamics::n_body_df_ephemeris, y1.size(), y1.data(), t0 + dt, data->eps, data->eps, &p);

        // calculate constraints

        // match point constraints
        std::vector<double> yff(6);
        std::vector<double> yfb(6);

        equation0.step(t0 + 0.5 * dt);
        equation0.get_y(0, 6, yff.data());

        equation1.step(t0 + 0.5 * dt);
        equation1.get_y(0, 6, yfb.data());

        for (int i = 0; i < 6; i++)
        {
            *result_ptr++ = yff[i] - yfb[i];
        }

        auto calc_fpa = [](double ra, double dec, double vra, double vdec)
        {
            return cos(dec) * cos(vdec) * cos(ra - vra) + sin(dec) * sin(vdec);
        };

        // flight path angle constraints
        *result_ptr++ = calc_fpa(x[5], x[6], x[8], x[9]);
        *result_ptr++ = calc_fpa(x[0], x[1], x[3], x[4]);

        // inclination constraints
        auto calc_h = [](double ra, double dec, double vra, double vdec)
        {
            Eigen::Vector3d r = { cos(ra) * cos(dec), sin(dec), sin(ra) * cos(dec) };
            Eigen::Vector3d v = { cos(vra) * cos(vdec), sin(vdec), sin(vra) * cos(vdec) };
            return r.cross(v);
        };

        double cosi_launch = data->n_launch.dot(calc_h(x[5], x[6], x[8], x[9]));
        double cosi_arrival = data->n_arrival.dot(calc_h(x[0], x[1], x[3], x[4]));

        *result_ptr++ = cosi_launch - cos(data->min_inclination_launch) + min_launch_inc_slack;
        *result_ptr++ = cosi_launch - cos(data->max_inclination_launch) - max_launch_inc_slack;

        *result_ptr++ = cosi_arrival - cos(data->min_inclination_arrival) + min_arrival_inc_slack;
        *result_ptr++ = cosi_arrival - cos(data->max_inclination_arrival) - max_arrival_inc_slack;

        // time constraints
        *result_ptr++ = dt - data->min_time - min_time_slack;
        *result_ptr++ = dt - data->max_time + max_time_slack;

        // calculate gradients
        if (grad)
        {
            constraint_numerical_gradient(m, n, x, grad, f_data, return_constraints);
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "error in constraint function:" << '\n';
        std::cerr << e.what() << '\n';
        throw e;
    }
}

double LunarFlightPlan::free_return_objective(unsigned n, const double* x, double* grad, void* f_data)
{
    try
    {
        FlightPlanData* data = reinterpret_cast<FlightPlanData*>(f_data);

        if (grad)
        {
            objective_numerical_gradient(n, x, grad, f_data, free_return_objective);
        }

        return x[2];
    }
    catch (const std::exception& e)
    {
        std::cerr << "error in objective function:" << '\n';
        std::cerr << e.what() << '\n';
        throw e;
    }
    
}

double LunarFlightPlan::leave_objective(unsigned n, const double* x, double* grad, void* f_data)
{
    try
    {
        FlightPlanData* data = reinterpret_cast<FlightPlanData*>(f_data);

        if (grad)
        {
            objective_numerical_gradient(n, x, grad, f_data, leave_objective);
        }

        return x[2] + x[7];
    }
    catch (const std::exception& e)
    {
        std::cerr << "error in objective function:" << '\n';
        std::cerr << e.what() << '\n';
        throw e;
    }

}

double LunarFlightPlan::return_objective(unsigned n, const double* x, double* grad, void* f_data)
{
    try
    {
        FlightPlanData* data = reinterpret_cast<FlightPlanData*>(f_data);

        if (grad)
        {
            objective_numerical_gradient(n, x, grad, f_data, return_objective);
        }

        return x[7];
    }
    catch (const std::exception& e)
    {
        std::cerr << "error in objective function:" << '\n';
        std::cerr << e.what() << '\n';
        throw e;
    }

}

void LunarFlightPlan::constraint_numerical_gradient( unsigned m, unsigned n, const double* x, double* grad, void* f_data,
    void(*func)(unsigned m, double* result, unsigned n, const double* x, double* grad, void* f_data))
{
    double eps = 1.49011611938476e-08;
    std::vector<double> base(m);
    std::vector<double> pert(m);
    std::vector<double> p(n);

    for (int i = 0; i < n; i++)
    {
        p[i] = x[i];
    }

    func(m, base.data(), n, x, NULL, f_data);

    double pold;

    for (int i = 0; i < n; i++)
    {
        if (i != 0)
        {
            p[i - 1] = pold;
        }
        pold = p[i];
        p[i] += eps;

        func(m, pert.data(), n, p.data(), NULL, f_data);

        for (int j = 0; j < m; j++)
        {
            grad[i + j * n] = (pert[j] - base[j]) / eps;
        }
    }
}

void LunarFlightPlan::objective_numerical_gradient(unsigned n, const double* x, double* grad,
    void* f_data, double(*func)(unsigned n, const double* x, double* grad, void* f_data))
{
    double eps = 1.49011611938476e-08;
    double base = func(n, x, NULL, f_data);
    std::vector<double> p(n);

    for (int i = 0; i < n; i++)
    {
        p[i] = x[i];
    }

    double pold;

    for (int i = 0; i < n; i++)
    {
        if (i != 0)
        {
            p[i - 1] = pold;
        }
        pold = p[i];
        p[i] += eps;

        grad[i] = (func(n, p.data(), NULL, f_data) - base) / eps;
    }
}
