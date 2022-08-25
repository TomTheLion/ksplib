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
    double dt = astrodynamics::pi / 6.0 * sqrt(pow(rm0.norm(), 3) / mu_earth);
    auto [rm1, vm1] = data_.ephemeris->get_position_velocity("Moon", data_.initial_time + dt);

    Eigen::Vector3d n = rm1.cross(vm1).normalized();

    Eigen::Vector3d r0 = -data_.rp_earth * rm1.normalized();
    Eigen::Vector3d v0 = sqrt(mu_earth * (2.0 / data_.rp_earth - 2.0 / rm1.norm())) * rm1.cross(n).normalized();

    Eigen::Vector3d r1 = data_.rp_moon * rm1.normalized();
    Eigen::Vector3d v1 = sqrt(mu_moon * (2.0 / data_.rp_moon - 2.0 / rm1.norm())) * rm1.cross(n).normalized();

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

        x_.push_back(0.5 * dt);
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
    int m = 14;

    if (data_.initial_orbit)
    {
        n -= 4;
        m -= 2;
    }

    if (data_.free_return)
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
    opt_.set_min_objective(objective, &data_);
    opt_.add_equality_mconstraint(free_return_constraints, &data_, tol);
    opt_.set_maxeval(max_eval);
    opt_.set_ftol_abs(eps_f);
    opt_.optimize(x_, minf);

    output_result(x_.data(), &data_);

    std::vector<double> result(m);

    free_return_constraints(m, result.data(), n, x_.data(), nullptr, &data_);

    std::cout << '\n';
    std::cout << '\n';

    for (int i = 0; i < m; i++)
    {
        std::cout << std::setprecision(17) << result[i] << '\n';
    }

    std::cout << '\n';
    std::cout << '\n';

    std::cout << opt_.last_optimize_result() << '\n';
    std::cout << opt_.last_optimum_value() << '\n';
    std::cout << opt_.get_numevals() << '\n';
}

LunarFlightPlan::Result LunarFlightPlan::output_result(const double* x, void* f_data)
{
    try
    {
        FlightPlanData* data = reinterpret_cast<FlightPlanData*>(f_data);

        const double* x_ptr = x;

        // parse inputs
        std::vector<double> y0 = cartesian_state(x_ptr, data->rp_earth);
        std::vector<double> y1 = cartesian_state(x_ptr);
        std::vector<double> y2 = cartesian_state(x_ptr, data->rp_earth);

        double t0 = *x_ptr++;
        double dt1 = *x_ptr++;
        double dt2 = *x_ptr++;

        auto [rm, vm] = data->ephemeris->get_position_velocity("Moon", t0 + dt1);

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

        // first leg
        for (int i = 0; i < 33; i++)
        {
            equation0.step(t0 + (dt1 + dt2) * pow((double)i / 32, 1));
            std::cout << std::setprecision(17)  << equation0.get_y(0) << ", " << equation0.get_y(1) << ", " << equation0.get_y(2) << '\n';
        }

        for (int i = 0; i < 33; i++)
        {
            equation1.step(t0 + dt1 - 0.5 * dt1 * (double)i / 32);
            std::cout << std::setprecision(17) << equation1.get_y(0) << ", " << equation1.get_y(1) << ", " << equation1.get_y(2) << '\n';
        }

        for (int i = 0; i < 33; i++)
        {
            Eigen::Vector3d rmt = data->ephemeris->get_position("Moon", t0 + dt1 - 0.5 * dt1 * (double)i / 32);
            std::cout << std::setprecision(17) << rmt(0) << ", " << rmt(1) << ", " << rmt(2) << '\n';
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "error in constraint function:" << '\n';
        std::cerr << e.what() << '\n';
        throw e;
    }

    return Result();
}

std::tuple<std::vector<double>, std::vector<double>> LunarFlightPlan::bounds()
{
    std::vector<double> lower_bounds;
    std::vector<double> upper_bounds;

    double mu_earth = data_.ephemeris->get_muk();
    double mu_moon = data_.ephemeris->get_mu("Moon");
    double d_scale = data_.ephemeris->get_distance_scale();
    double v_scale = data_.ephemeris->get_velocity_scale();

    auto push_back_state_bounds = [d_scale, v_scale](std::vector<double>& lower_bounds, std::vector<double>& upper_bounds, double mu, double rp, bool initial_orbit, bool radius)
    {
        if (!initial_orbit)
        {
            if (radius)
            {
                lower_bounds.push_back(1731100.0 / d_scale);
                upper_bounds.push_back(20000000.0 / d_scale);
            }

            lower_bounds.push_back(-2.0 * astrodynamics::pi);
            lower_bounds.push_back(-2.0 * astrodynamics::pi);

            upper_bounds.push_back(2.0 * astrodynamics::pi);
            upper_bounds.push_back(2.0 * astrodynamics::pi);
        }

        lower_bounds.push_back(sqrt(mu / rp));
        lower_bounds.push_back(-2.0 * astrodynamics::pi);
        lower_bounds.push_back(-2.0 * astrodynamics::pi);

        upper_bounds.push_back(sqrt(4.0 * mu / rp));
        upper_bounds.push_back(2.0 * astrodynamics::pi);
        upper_bounds.push_back(2.0 * astrodynamics::pi);
    };

    if (data_.free_return)
    {
        if (data_.initial_orbit)
        {
            push_back_state_bounds(lower_bounds, upper_bounds, mu_earth, data_.rp_earth, true, false);
        }
        else
        {
            push_back_state_bounds(lower_bounds, upper_bounds, mu_earth, data_.rp_earth, false, false);
        }

        push_back_state_bounds(lower_bounds, upper_bounds, mu_moon, data_.rp_moon, false, true);
        push_back_state_bounds(lower_bounds, upper_bounds, mu_earth, data_.rp_earth, false, false);

        if (data_.initial_orbit)
        {
            lower_bounds.push_back(0.0);
            upper_bounds.push_back(2.0 * astrodynamics::pi * sqrt(pow(data_.initial_position.norm(), 3) / data_.ephemeris->get_muk()));
        }
        else
        {
            lower_bounds.push_back(data_.initial_time);
            upper_bounds.push_back(HUGE_VAL);
        }

        lower_bounds.push_back(day_);
        lower_bounds.push_back(day_);
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

        if (!data_.initial_orbit)
        {
            lower_bounds.push_back(0.0);
            lower_bounds.push_back(0.0);
            upper_bounds.push_back(HUGE_VAL);
            upper_bounds.push_back(HUGE_VAL);
        }
    }
    else
    {
        if (data_.initial_orbit)
        {
            push_back_state_bounds(lower_bounds, upper_bounds, mu_earth, data_.rp_earth, true, false);
        }
        else
        {
            push_back_state_bounds(lower_bounds, upper_bounds, mu_earth, data_.rp_earth, false, false);
        }

        push_back_state_bounds(lower_bounds, upper_bounds, mu_moon, data_.rp_moon, false, false);

        if (data_.initial_orbit)
        {
            lower_bounds.push_back(0.0);
            upper_bounds.push_back(2.0 * astrodynamics::pi * sqrt(pow(data_.initial_position.norm(), 3) / data_.ephemeris->get_muk()));
        }
        else
        {
            lower_bounds.push_back(data_.initial_time);
            upper_bounds.push_back(HUGE_VAL);
        }

        lower_bounds.push_back(day_);
        lower_bounds.push_back(0.0);
        lower_bounds.push_back(0.0);
        lower_bounds.push_back(0.0);
        lower_bounds.push_back(0.0);

        upper_bounds.push_back(HUGE_VAL);
        upper_bounds.push_back(HUGE_VAL);
        upper_bounds.push_back(HUGE_VAL);
        upper_bounds.push_back(HUGE_VAL);
        upper_bounds.push_back(HUGE_VAL);

        if (!data_.initial_orbit)
        {
            lower_bounds.push_back(0.0);
            lower_bounds.push_back(0.0);
            upper_bounds.push_back(HUGE_VAL);
            upper_bounds.push_back(HUGE_VAL);
        }
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

        double t0 = *x_ptr++;
        double dt1 = *x_ptr++;
        double dt2 = *x_ptr++;

        auto [rm, vm] = data->ephemeris->get_position_velocity("Moon", t0 + dt1);

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

        // first leg
        //for (int i = 0; i < 33; i++)
        //{
        //    equation0.step(t0 + 0.5 * dt1 * pow((double)i / 32, 2));
        //    std::cout << std::setprecision(17)  << equation0.get_y(0) << ", " << equation0.get_y(1) << ", " << equation0.get_y(2) << '\n';
        //}

        //for (int i = 0; i < 33; i++)
        //{
        //    equation1.step(t0 + dt1 - 0.5 * dt1 * (double)i / 32);
        //    std::cout << std::setprecision(17) << equation1.get_y(0) << ", " << equation1.get_y(1) << ", " << equation1.get_y(2) << '\n';
        //}

        //for (int i = 0; i < 33; i++)
        //{
        //    Eigen::Vector3d rmt = data->ephemeris->get_position("Moon", t0 + dt1 - 0.5 * dt1 * (double)i / 32);
        //    std::cout << std::setprecision(17) << rmt(0) << ", " << rmt(1) << ", " << rmt(2) << '\n';
        //}


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
        double cosi_arrival = data->n_arrival.dot(calc_h(x[6], x[7], x[9], x[10]));

        *result_ptr++ = cosi_launch - cos(data->min_inclination_launch) + min_launch_inc_slack;
        *result_ptr++ = cosi_launch - cos(data->max_inclination_launch) - max_launch_inc_slack;

        *result_ptr++ = cosi_arrival - cos(data->min_inclination_arrival) + min_arrival_inc_slack;
        *result_ptr++ = cosi_arrival - cos(data->max_inclination_arrival) - max_arrival_inc_slack;

        // time constraints
        *result_ptr++ = dt1 + dt2 - data->min_time - min_time_slack;
        *result_ptr++ = dt1 + dt2 - data->max_time + max_time_slack;

        std::vector<double> rr;
        for (int i = 0; i < m; i++)
        {
            rr.push_back(result[i]);
        }

        // calculate gradients
        if (grad)
        {
            constraint_numerical_gradient(m, n, x, grad, f_data, free_return_constraints);
            //for (int j = 0; j < m; j++)
            //{
            //    for (int i = 0; i < n; i++)
            //    {

            //        std::cout << std::setprecision(17) << grad[i + n * j] << ", ";

            //    }
            //    std::cout << '\n';
            //}

            //int sldkfj = 231;
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "error in constraint function:" << '\n';
        std::cerr << e.what() << '\n';
        throw e;
    }
}

double LunarFlightPlan::objective(unsigned n, const double* x, double* grad, void* f_data)
{
    try
    {
        FlightPlanData* data = reinterpret_cast<FlightPlanData*>(f_data);

        if (grad)
        {
            objective_numerical_gradient(n, x, grad, f_data, objective);
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
