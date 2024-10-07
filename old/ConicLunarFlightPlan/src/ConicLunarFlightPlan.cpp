#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "ConicLunarFlightPlan.h"

ConicLunarFlightPlan::ConicLunarFlightPlan(astrodynamics::ConicBody& planet, astrodynamics::ConicBody& moon)
{
    data_.planet = &planet;
    data_.moon = &moon;

    data_.min_time = 1e4;
    data_.max_time = 1e6;
    data_.min_inclination_launch = 0.0;
    data_.max_inclination_launch = astrodynamics::pi;
    data_.n_launch_direction = 1.0;
    data_.n_launch = -Eigen::Vector3d::UnitY();
    data_.n_launch_plane = -Eigen::Vector3d::UnitY();
    data_.min_inclination_arrival = 0.0;
    data_.max_inclination_arrival = astrodynamics::pi;
    data_.n_arrival = -Eigen::Vector3d::UnitY();
}

ConicLunarFlightPlan::~ConicLunarFlightPlan()
{

}

//
// constraint functions
//

void ConicLunarFlightPlan::set_mission(double initial_time, TrajectoryMode mode, double rp_planet, double rp_moon, double e_moon)
{
    data_.mode = mode;
    data_.initial_time = initial_time;
    data_.rp_planet = rp_planet;
    data_.rp_moon = rp_moon;
    e_moon_ = e_moon;
}

void ConicLunarFlightPlan::add_min_flight_time_constraint(double min_time)
{
    data_.min_time = min_time;
}

void ConicLunarFlightPlan::add_max_flight_time_constraint(double max_time)
{
    data_.max_time = max_time;
}

void ConicLunarFlightPlan::add_inclination_constraint(bool launch, double min, double max, Eigen::Vector3d n)
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

void ConicLunarFlightPlan::add_launch_plane_constraint(double direction, Eigen::Vector3d n)
{
    data_.n_launch_direction = direction;
    data_.n_launch_plane = n;
}

void ConicLunarFlightPlan::init_model()
{
    double mu_planet = data_.planet->mu;
    double mu_moon = data_.moon->mu;

    Eigen::Vector3d rm = data_.moon->orbit->get_position(data_.initial_time);
    Eigen::Vector3d vm = data_.moon->orbit->get_velocity(data_.initial_time);

    double dt = sqrt(0.5 * pow(rm.norm(), 3) / mu_planet);
    if (data_.mode != TrajectoryMode::RETURN)
    {
        rm = data_.moon->orbit->get_position(data_.initial_time + dt);
        vm = data_.moon->orbit->get_velocity(data_.initial_time + dt);
    }

    Eigen::Vector3d n = rm.cross(vm).normalized();

    double min_e_planet = (rm.norm() - data_.rp_planet) / (rm.norm() + data_.rp_planet);

    Eigen::Vector3d r0 = -data_.rp_planet * rm.normalized();
    Eigen::Vector3d v0 = sqrt((1.0 + min_e_planet) * mu_planet / data_.rp_planet) * rm.cross(n).normalized();

    Eigen::Vector3d r1, v1;

    r1 = data_.rp_moon * rm.normalized();
    v1 = -sqrt((1.0 + e_moon_) * mu_moon / data_.rp_moon) * rm.cross(n).normalized();

    v0 = Eigen::AngleAxisd(data_.n_launch_direction * data_.min_inclination_launch, rm.normalized()) * v0;

    switch (data_.mode)
    {
    case TrajectoryMode::FREE_RETURN:
        if (data_.min_inclination_launch < astrodynamics::pi / 2.0)
        {
            v1 *= -1.0;
        }
        else
        {
            v0 *= -1.0;
        }
        break;
    case TrajectoryMode::LEAVE:
        if (data_.min_inclination_launch > astrodynamics::pi / 2.0)
        {
            v0 *= -1.0;
        }
        if (data_.min_inclination_arrival > astrodynamics::pi / 2.0)
        {
            v1 *= -1.0;
        }
        break;
    case TrajectoryMode::RETURN:
        if (data_.min_inclination_launch > astrodynamics::pi / 2.0)
        {
            v1 *= -1.0;
        }
        if (data_.min_inclination_arrival > astrodynamics::pi / 2.0)
        {
            v0 *= -1.0;
        }
        break;
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
        x_.push_back(2.0 * dt - data_.min_time);
        x_.push_back(data_.max_time - 2.0 * dt);
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
        x_.push_back(dt - data_.min_time);
        x_.push_back(data_.max_time - dt);
    }
}

void ConicLunarFlightPlan::run_model(int max_eval, double eps, double eps_t, double eps_x)
{
    // calculate number of constraints and decision variables
    int n = 18;
    int m = 15;

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

// set conic solution
void ConicLunarFlightPlan::set_conic_solution(std::vector<double> x)
{
    x_ = x;
}

ConicLunarFlightPlan::Result ConicLunarFlightPlan::output_result(double eps)
{
    Result result;

    int n = 18;
    int m = 15;

    if (data_.mode == TrajectoryMode::FREE_RETURN)
    {
        n += 7;
        m += 7;
    }

    data_.eps = eps;

    result.nlopt_constraints.resize(m);
    result.r.resize(3);
    result.v.resize(3);
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

    const double* x_ptr = x_.data();

    auto output = [this, &result](Eigen::Vector3d r, Eigen::Vector3d v, double mu, int leg, double ti, double tf, double tref, bool moon)
    {
        int steps = 20;

        for (int i = 0; i <= steps; i++)
        {
            double dt = (tf - ti) * static_cast<double>(i) / steps;
            double t = ti + dt;
            result.time.push_back(t);
            result.leg.push_back(leg);

            Eigen::Vector3d rm = data_.moon->orbit->get_position(t);
            Eigen::Vector3d vm = data_.moon->orbit->get_velocity(t);

            auto [rf, vf] = astrodynamics::kepler_s(r, v, t - tref, mu, data_.eps);

            if (moon)
            {
                rf += rm;
                vf += vm;
            }

            result.r[0].push_back(rf[0]);
            result.r[1].push_back(rf[1]);
            result.r[2].push_back(rf[2]);
            result.v[0].push_back(vf[0]);
            result.v[1].push_back(vf[1]);
            result.v[2].push_back(vf[2]);

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
            auto [r0, v0] = cartesian_state(x_ptr, data_.rp_planet);
            auto [r1, v1] = cartesian_state(x_ptr, 0.0);
            auto [r2, v2] = cartesian_state(x_ptr, data_.rp_planet);

            double t0 = data_.initial_time;
            double dt1 = *x_ptr++;
            double dt2 = *x_ptr++;
            double dtsoi = astrodynamics::hyperbolic_orbit_time_at_distance(r1, v1, data_.moon->soi, data_.moon->mu);

            output(r0, v0, data_.planet->mu, 0, t0, t0 + dt1 - dtsoi, t0, false);
            output(r1, v1, data_.moon->mu, 1, t0 + dt1 - dtsoi, t0 + dt1, t0 + dt1, true);
            output(r1, v1, data_.moon->mu, 2, t0 + dt1, t0 + dt1 + dtsoi, t0 + dt1, true);
            output(r2, v2, data_.planet->mu, 3, t0 + dt1 + dtsoi, t0 + dt1 + dt2, t0 + dt1 + dt2, false);
            break;
        }
    case TrajectoryMode::LEAVE:
        {
            leave_constraints(m, result.nlopt_constraints.data(), n, x_.data(), NULL, &data_);

            // parse inputs
            auto [r0, v0] = cartesian_state(x_ptr, data_.rp_planet);
            auto [r1, v1] = cartesian_state(x_ptr, data_.rp_moon);

            double t0 = data_.initial_time;
            double dt = *x_ptr++;
            double dtsoi = astrodynamics::hyperbolic_orbit_time_at_distance(r1, v1, data_.moon->soi, data_.moon->mu);

            output(r0, v0, data_.planet->mu, 0, t0, t0 + dt - dtsoi, t0, false);
            output(r1, v1, data_.moon->mu, 1, t0 + dt - dtsoi, t0 + dt, t0 + dt, true);

            break;
        }
    case TrajectoryMode::RETURN:
        {
            return_constraints(m, result.nlopt_constraints.data(), n, x_.data(), NULL, &data_);

            // parse inputs
            auto [r1, v1] = cartesian_state(x_ptr, data_.rp_planet);
            auto [r0, v0] = cartesian_state(x_ptr, data_.rp_moon);

            double t0 = data_.initial_time;
            double dt = *x_ptr++;
            double dtsoi = astrodynamics::hyperbolic_orbit_time_at_distance(r0, v0, data_.moon->soi, data_.moon->mu);

            output(r0, v0, data_.moon->mu, 0, t0, t0 + dtsoi, t0, true);
            output(r1, v1, data_.planet->mu, 1, t0 + dtsoi, t0 + dt, t0 + dt, false);
            break;
        }
    }

    return result;
}

std::tuple<std::vector<double>, std::vector<double>> ConicLunarFlightPlan::bounds()
{
    std::vector<double> lower_bounds;
    std::vector<double> upper_bounds;

    double mu_planet = data_.planet->mu;
    double mu_moon = data_.moon->mu;
    double moon_radius = data_.moon->radius;
    double moon_soi_radius = data_.moon->soi;

    double rm = data_.moon->orbit->get_position(data_.initial_time).norm();

    double min_e_moon = (moon_soi_radius - data_.rp_moon) / (moon_soi_radius + data_.rp_moon);
    double min_e_planet = (rm - data_.rp_planet) / (rm + data_.rp_planet);
 
    auto push_back_state_bounds = [moon_radius, moon_soi_radius]
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
        push_back_state_bounds(lower_bounds, upper_bounds, mu_planet, data_.rp_planet, min_e_planet, false);
        push_back_state_bounds(lower_bounds, upper_bounds, mu_moon, data_.rp_moon, min_e_moon, true);
        push_back_state_bounds(lower_bounds, upper_bounds, mu_planet, data_.rp_planet, min_e_planet, false);

        lower_bounds.push_back(data_.min_time);
        lower_bounds.push_back(data_.min_time);
        lower_bounds.push_back(0.0);
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
        upper_bounds.push_back(HUGE_VAL);
    }
    else
    {
        push_back_state_bounds(lower_bounds, upper_bounds, mu_planet, data_.rp_planet, min_e_planet, false);
        push_back_state_bounds(lower_bounds, upper_bounds, mu_moon, data_.rp_moon, min_e_moon, false);

        lower_bounds.push_back(data_.min_time);
        lower_bounds.push_back(0.0);
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

    return { lower_bounds, upper_bounds };
}

std::tuple<Eigen::Vector3d, Eigen::Vector3d> ConicLunarFlightPlan::cartesian_state(const double*& x_ptr, double rho)
{
    Eigen::Vector3d r;
    Eigen::Vector3d v;

    if (rho == 0)
    {
        rho = *x_ptr++;
    }

    double ra = *x_ptr++;
    double dec = *x_ptr++;
    double vrho = *x_ptr++;
    double vra = *x_ptr++;
    double vdec = *x_ptr++;

    r <<
        rho * cos(ra) * cos(dec),
        rho * sin(dec),
        rho * sin(ra) * cos(dec);
    v <<
        vrho * cos(vra) * cos(vdec),
        vrho * sin(vdec),
        vrho * sin(vra) * cos(vdec);

    return { r, v };
}

void ConicLunarFlightPlan::free_return_constraints(unsigned m, double* result, unsigned n, const double* x, double* grad, void* f_data)
{
    try
    {
        FlightPlanData* data = reinterpret_cast<FlightPlanData*>(f_data);

        const double* x_ptr = x;
        double* result_ptr = result;

        // parse inputs
        auto [r0, v0] = cartesian_state(x_ptr, data->rp_planet);
        auto [r1, v1] = cartesian_state(x_ptr, 0.0);
        auto [r2, v2] = cartesian_state(x_ptr, data->rp_planet);

        double t0 = data->initial_time;
        double dt1 = *x_ptr++;
        double dt2 = *x_ptr++;

        double min_launch_inc_slack = *x_ptr++;
        double max_launch_inc_slack = *x_ptr++;
        double launch_plane_slack = *x_ptr++;
        double min_arrival_inc_slack = *x_ptr++;
        double max_arrival_inc_slack = *x_ptr++;
        double min_time_slack = *x_ptr++;
        double max_time_slack = *x_ptr++;

        // propagate
        double dtsoi = astrodynamics::hyperbolic_orbit_time_at_distance(r1, v1, data->moon->soi, data->moon->mu);

        auto [r0f, v0f] = astrodynamics::kepler_s(r0, v0, dt1 - dtsoi, data->planet->mu, data->eps);
        auto [r1b, v1b] = astrodynamics::kepler_s(r1, v1, -dtsoi, data->moon->mu, data->eps);

        auto [r1f, v1f] = astrodynamics::kepler_s(r1, v1, dtsoi, data->moon->mu, data->eps);
        auto [r2b, v2b] = astrodynamics::kepler_s(r2, v2, dtsoi - dt2, data->planet->mu, data->eps);

        // calculate constraints

        // match point constraints

        Eigen::Vector3d rm1 = data->moon->orbit->get_position(t0 + dt1 - dtsoi);
        Eigen::Vector3d vm1 = data->moon->orbit->get_velocity(t0 + dt1 - dtsoi);

        Eigen::Vector3d dr1 = r0f - (r1b + rm1);
        Eigen::Vector3d dv1 = v0f - (v1b + vm1);

        Eigen::Vector3d rm2 = data->moon->orbit->get_position(t0 + dt1 + dtsoi);
        Eigen::Vector3d vm2 = data->moon->orbit->get_velocity(t0 + dt1 + dtsoi);

        Eigen::Vector3d dr2 = (r1f + rm2) - r2b;
        Eigen::Vector3d dv2 = (v1f + vm2) - v2b;

        *result_ptr++ = dr1(0) / data->moon->orbit->get_distance_scale();
        *result_ptr++ = dr1(1) / data->moon->orbit->get_distance_scale();
        *result_ptr++ = dr1(2) / data->moon->orbit->get_distance_scale();
        *result_ptr++ = dv1(0) / data->moon->orbit->get_velocity_scale();
        *result_ptr++ = dv1(1) / data->moon->orbit->get_velocity_scale();
        *result_ptr++ = dv1(2) / data->moon->orbit->get_velocity_scale();

        *result_ptr++ = dr2(0) / data->moon->orbit->get_distance_scale();
        *result_ptr++ = dr2(1) / data->moon->orbit->get_distance_scale();
        *result_ptr++ = dr2(2) / data->moon->orbit->get_distance_scale();
        *result_ptr++ = dv2(0) / data->moon->orbit->get_velocity_scale();
        *result_ptr++ = dv2(1) / data->moon->orbit->get_velocity_scale();
        *result_ptr++ = dv2(2) / data->moon->orbit->get_velocity_scale();

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

        *result_ptr++ = -data->n_launch_plane.dot(v0.normalized()) * data->n_launch_direction - launch_plane_slack;

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

void ConicLunarFlightPlan::leave_constraints(unsigned m, double* result, unsigned n, const double* x, double* grad, void* f_data)
{
    try
    {
        FlightPlanData* data = reinterpret_cast<FlightPlanData*>(f_data);

        const double* x_ptr = x;
        double* result_ptr = result;

        // parse inputs
        auto [r0, v0] = cartesian_state(x_ptr, data->rp_planet);
        auto [r1, v1] = cartesian_state(x_ptr, data->rp_moon);

        double t0 = data->initial_time;
        double dt = *x_ptr++;
        
        double min_launch_inc_slack = *x_ptr++;
        double max_launch_inc_slack = *x_ptr++;
        double launch_plane_slack = *x_ptr++;
        double min_arrival_inc_slack = *x_ptr++;
        double max_arrival_inc_slack = *x_ptr++;
        double min_time_slack = *x_ptr++;
        double max_time_slack = *x_ptr++;

        // propagate
        double dtsoi = astrodynamics::hyperbolic_orbit_time_at_distance(r1, v1, data->moon->soi, data->moon->mu);

        auto [r0f, v0f] = astrodynamics::kepler_s(r0, v0, dt - dtsoi, data->planet->mu, data->eps);
        auto [r1b, v1b] = astrodynamics::kepler_s(r1, v1, -dtsoi, data->moon->mu, data->eps);

        // calculate constraints

        // match point constraints

        Eigen::Vector3d rm = data->moon->orbit->get_position(t0 + dt - dtsoi);
        Eigen::Vector3d vm = data->moon->orbit->get_velocity(t0 + dt - dtsoi);

        Eigen::Vector3d dr = r0f - (r1b + rm);
        Eigen::Vector3d dv = v0f - (v1b + vm);

        *result_ptr++ = dr(0) / data->moon->orbit->get_distance_scale();
        *result_ptr++ = dr(1) / data->moon->orbit->get_distance_scale();
        *result_ptr++ = dr(2) / data->moon->orbit->get_distance_scale();
        *result_ptr++ = dv(0) / data->moon->orbit->get_velocity_scale();
        *result_ptr++ = dv(1) / data->moon->orbit->get_velocity_scale();
        *result_ptr++ = dv(2) / data->moon->orbit->get_velocity_scale();

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

        *result_ptr++ = -data->n_launch_plane.dot(v0.normalized()) * data->n_launch_direction - launch_plane_slack;

        *result_ptr++ = cosi_arrival - cos(data->min_inclination_arrival) + min_arrival_inc_slack;
        *result_ptr++ = cosi_arrival - cos(data->max_inclination_arrival) - max_arrival_inc_slack;

        // time constraints
        *result_ptr++ = (dt - data->min_time - min_time_slack);
        *result_ptr++ = (dt - data->max_time + max_time_slack);

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

void ConicLunarFlightPlan::return_constraints(unsigned m, double* result, unsigned n, const double* x, double* grad, void* f_data)
{
    try
    {
        FlightPlanData* data = reinterpret_cast<FlightPlanData*>(f_data);

        const double* x_ptr = x;
        double* result_ptr = result;

        // parse inputs
        auto [r1, v1] = cartesian_state(x_ptr, data->rp_planet);
        auto [r0, v0] = cartesian_state(x_ptr, data->rp_moon);

        double t0 = data->initial_time;
        double dt = *x_ptr++;

        double min_launch_inc_slack = *x_ptr++;
        double max_launch_inc_slack = *x_ptr++;
        double launch_plane_slack = *x_ptr++;
        double min_arrival_inc_slack = *x_ptr++;
        double max_arrival_inc_slack = *x_ptr++;
        double min_time_slack = *x_ptr++;
        double max_time_slack = *x_ptr++;

        // integrate
        double dtsoi = astrodynamics::hyperbolic_orbit_time_at_distance(r0, v0, data->moon->soi, data->moon->mu);

        auto [r0f, v0f] = astrodynamics::kepler_s(r0, v0, dtsoi, data->moon->mu, data->eps);
        auto [r1b, v1b] = astrodynamics::kepler_s(r1, v1, dtsoi - dt, data->planet->mu, data->eps);

        // calculate constraints

        // match point constraints

        Eigen::Vector3d rm = data->moon->orbit->get_position(t0 + dtsoi);
        Eigen::Vector3d vm = data->moon->orbit->get_velocity(t0 + dtsoi);

        Eigen::Vector3d dr = (r0f + rm) - r1b;
        Eigen::Vector3d dv = (v0f + vm) - v1b;

        *result_ptr++ = dr(0) / data->moon->orbit->get_distance_scale();
        *result_ptr++ = dr(1) / data->moon->orbit->get_distance_scale();
        *result_ptr++ = dr(2) / data->moon->orbit->get_distance_scale();
        *result_ptr++ = dv(0) / data->moon->orbit->get_velocity_scale();
        *result_ptr++ = dv(1) / data->moon->orbit->get_velocity_scale();
        *result_ptr++ = dv(2) / data->moon->orbit->get_velocity_scale();

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

        *result_ptr++ = -data->n_launch_plane.dot(v0.normalized()) * data->n_launch_direction - launch_plane_slack;

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

double ConicLunarFlightPlan::free_return_objective(unsigned n, const double* x, double* grad, void* f_data)
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

double ConicLunarFlightPlan::leave_objective(unsigned n, const double* x, double* grad, void* f_data)
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

double ConicLunarFlightPlan::return_objective(unsigned n, const double* x, double* grad, void* f_data)
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

void ConicLunarFlightPlan::constraint_numerical_gradient( unsigned m, unsigned n, const double* x, double* grad, void* f_data,
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

void ConicLunarFlightPlan::objective_numerical_gradient(unsigned n, const double* x, double* grad,
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
