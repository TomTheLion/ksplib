#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>

#include <Eigen/Dense>

#include "py_lib.h"

#include "Equation.h"
#include "Spl.h"


double* py_arr_ptr(py::object obj)
{
    return static_cast<double*>(obj.cast<py::array_t<double>>().request().ptr);
}

int py_arr_size(py::object obj)
{
    return obj.cast<py::array_t<double>>().request().size;
}

py::array_t<double> py_array_copy(py::object obj)
{
    py::array_t<double> arr_copy = py::array_t<double>(obj.cast<py::array_t<double>>().request());
    return arr_copy;
}

std::vector<double> py_std_vector(py::object obj)
{
    size_t size = py_arr_size(obj);
    double* data = py_arr_ptr(obj);
    std::vector<double> arr_copy = std::vector<double>(data, data + size);
    return arr_copy;
}

Eigen::Vector3d py_vector3d(py::object obj)
{
    return Eigen::Map<Eigen::Vector3d>(py_arr_ptr(obj.cast<py::array_t<double>>()));
}

Eigen::Vector3d py_vector3d(py::object obj, int n)
{
    return Eigen::Map<Eigen::Vector3d>(py_arr_ptr(obj.cast<py::array_t<double>>()) + n);
}

Spl py_spl(py::object obj)
{
    py::tuple tup = obj;
    Spl s = { py_arr_ptr(tup[0]), py_arr_ptr(tup[1]), py_arr_size(tup[0]), tup[2].cast<int>() };
    return s;
}

Spl py_bspl(py::object obj) {
    py::tuple tup = obj;
    Spl s = {
        py_arr_ptr(tup[0]), py_arr_ptr(tup[1]), py_arr_ptr(tup[2]),
        py_arr_size(tup[0]), py_arr_size(tup[1]),
        tup[3].cast<int>(), tup[4].cast<int>() };
    return s;
}

void py_copy_array(std::vector<double> x, py::array_t<double> py_x)
{
    double* data = py_x.mutable_data();
    std::copy(x.begin(), x.end(), data);
}

template<typename T>
py::list py_copy_list(std::vector<T> x)
{
    py::list py_list;

    for (auto& v : x) py_list.append(v);

    return py_list;
}

template<typename T>
py::array_t<T> py_copy_array(std::vector<T> x)
{
    int l = x.size();
    py::array_t<double> py_x({ 1, l });
    py_x = py_x.reshape({ l });

    double* py_mx = py_x.mutable_data();

    for (int i = 0; i < l; i++)
    {
        py_mx[i] = x[i];
    }

    return py_x;
}

//template<typename T>
//py::array_t<T, py::array::c_style> py_copy_array(std::vector<T> x)
//{
//    int l = x.size();
//
//    py::array_t<T, py::array::c_style> py_x({ l });
//    auto mx = py_x.mutable_unchecked();
//
//    for (int i = 0; i < l; i++)
//    {
//        mx(i) = x[i];
//    }
//
//    return py_x;
//}

py::array_t<double, py::array::c_style> py_copy_array(std::vector<std::vector<double>> x)
{
    size_t l = x.size();
    size_t m = x[0].size();

    py::array_t<double, py::array::c_style> py_x({ l, m });
    auto mx = py_x.mutable_unchecked();

    for (int i = 0; i < l; i++)
    {
        for (int j = 0; j < m; j++)
        {
            mx(i, j) = x[i][j];
        }
    }

    return py_x;
}

py::array_t<double, py::array::c_style> py_copy_array(std::vector<std::vector<std::vector<double>>> x)
{
    size_t l = x.size();
    size_t m = x[0].size();
    size_t n = x[0][0].size();
    py::array_t<double, py::array::c_style> py_x({ l, m, n });
    auto mx = py_x.mutable_unchecked();

    for (int i = 0; i < l; i++)
    {
        for (int j = 0; j < m; j++)
        {
            for (int k = 0; k < n; k++)
            {
                mx(i, j, k) = x[i][j][k];
            }
        }
    }

    return py_x;
}

namespace kerbal_guidance_system
{
    struct NDim
    {
    public:
        NDim();

        NDim(double distance, double velocity, double acceleration, double time, double mass)
            : distance_(distance), velocity_(velocity), acceleration_(acceleration), time_(time), mass_(mass) {}

        template <typename T> T distance_to(T x) { return x / distance_; }
        template <typename T> T distance_from(T x) { return x * distance_; }

        template <typename T> T velocity_to(T x) { return x / velocity_; }
        template <typename T> T velocity_from(T x) { return x * velocity_; }

        template <typename T> T acceleration_to(T x) { return x / acceleration_; }
        template <typename T> T acceleration_from(T x) { return x * acceleration_; }

        template <typename T> T time_to(T x) { return x / time_; }
        template <typename T> T time_from(T x) { return x * time_; }

        template <typename T> T mass_to(T x) { return x / mass_; }
        template <typename T> T mass_from(T x) { return x * mass_; }

        template <typename T> T force_to(T x) { return x / (mass_ * acceleration_); }
        template <typename T> T force_from(T x) { return x * (mass_ * acceleration_); }

        template <typename T> T rate_to(T x) { return x * time_; }
        template <typename T> T rate_from(T x) { return x / time_; }

        template <typename T> T mass_rate_to(T x) { return x / (mass_ / time_); }
        template <typename T> T mass_rate_from(T x) { return x * (mass_ / time_); }

        void state_to(double& t, std::vector<double>& y) const
        {
            t /= time_;
            y[0] /= distance_;
            y[1] /= distance_;
            y[2] /= distance_;
            y[3] /= velocity_;
            y[4] /= velocity_;
            y[5] /= velocity_;
            y[6] /= mass_;
        }

        void state_from(double& t, std::vector<double>& y) const
        {
            t *= time_;
            y[0] *= distance_;
            y[1] *= distance_;
            y[2] *= distance_;
            y[3] *= velocity_;
            y[4] *= velocity_;
            y[5] *= velocity_;
            y[6] *= mass_;
        }

    private:
        double distance_;
        double velocity_;
        double acceleration_;
        double time_;
        double mass_;
    };

    struct DragModel
    {
        Spl spl_pressure;
        Spl spl_density;
        Spl spl_drag;
        Spl spl_drag_mul;

        double eval(double r, double v)
        {
            double pressure = exp(spl_pressure.eval(r));
            double density = exp(spl_density.eval(r));
            double mach = v / sqrt(1.4 * pressure / density);
            double drag = spl_drag.eval(mach) * spl_drag_mul.eval(density * v) * density * v * v;
            return drag;
        }
    };

    struct Derivatives
    {
        double mass;
        Eigen::Vector3d drag;
        Eigen::Vector3d thrust;
        Eigen::Vector3d gravity;
        Eigen::Vector3d rotation;
    };

    struct AtmParams
    {
        NDim ndim;
        DragModel drag_model;
        Derivatives derivatives;
        double initial_angle;
        Eigen::Vector3d angular_velocity;
        double a_limit;
        double azimuth;
        double pitch_time;
        double pitch_duration;
        double pitch_rate;
        double switch_altitude;
        Spl thrust;
        double mass_rate;
    };

    struct VacParams
    {
        NDim ndim;
        Derivatives derivatives;
        double a_limit;
        double tg;
        Eigen::Vector3d lambda_i;
        Eigen::Vector3d lambda_dot_i;
        double final_time;
        double thrust;
        double mass_rate;
        double final_velocity;
    };

    static double safe_acos(double x)
    {
        if (x > 1)
        {
            return 0.0;
        }
        else if (x < -1)
        {
            return acos(-1.0);
        }
        else
        {
            return acos(x);
        }
    }

    static void atm_acceleration(double t, double y[], AtmParams* p)
    {
        Eigen::Map<Eigen::Vector3d> r(y);
        Eigen::Map<Eigen::Vector3d> v(y + 3);
        double m = y[6];

        Eigen::Vector3d attitude = r.normalized();
        double pitch = 0.0;
        if (t > p->pitch_time)
        {
            pitch = std::max(safe_acos(r.dot(v) / r.norm() / v.norm()), p->pitch_rate * std::min(t - p->pitch_time, p->pitch_duration));
            Eigen::Vector3d axis = Eigen::AngleAxisd(p->azimuth, r.normalized()) * Eigen::Vector3d::UnitY().cross(r).cross(r).normalized();
            attitude = Eigen::AngleAxisd(pitch, axis) * attitude;
        }

         // check angle between attitude and surface velocity...


        double rnorm = p->ndim.distance_from(r.norm());
        double vnorm = p->ndim.velocity_from(v.norm());
        double drag = p->ndim.force_to(p->drag_model.eval(rnorm, vnorm));
        double thrust = p->ndim.force_to(p->thrust.eval(rnorm));
        double throttle = p->a_limit ? std::min(1.0, p->a_limit / (thrust / m)) : 1.0;

        p->derivatives.mass = throttle * p->mass_rate;
        p->derivatives.drag = -v.normalized() * drag / m;
        p->derivatives.thrust = attitude * throttle * thrust / m;
        p->derivatives.gravity = -r / pow(r.norm(), 3);
        p->derivatives.rotation = -2 * p->angular_velocity.cross(v) - p->angular_velocity.cross(p->angular_velocity.cross(r));
    }

    static void vac_acceleration(double t, double y[], VacParams* p)
    {
        Eigen::Map<Eigen::Vector3d> r(y);
        Eigen::Map<Eigen::Vector3d> v(y + 3);
        double m = y[6];

        double a_thrust_mag = p->thrust / m;
        double throttle = p->a_limit ? std::min(1.0, p->a_limit / a_thrust_mag) : 1.0;
        Eigen::Vector3d lambda = p->lambda_i * cos(t - p->tg) + p->lambda_dot_i * sin(t - p->tg);

        p->derivatives.mass = throttle * p->mass_rate;
        p->derivatives.thrust = throttle * a_thrust_mag * lambda.normalized();
        p->derivatives.gravity = -r / pow(r.norm(), 3);
    }

    static void atm_state(double t, double y[], double yp[], void* params)
    {
        AtmParams* p = reinterpret_cast<AtmParams*>(params);
        atm_acceleration(t, y, p);
        Eigen::Vector3d a = p->derivatives.drag + p->derivatives.thrust + p->derivatives.gravity + p->derivatives.rotation;

        yp[0] = y[3];
        yp[1] = y[4];
        yp[2] = y[5];
        yp[3] = a[0];
        yp[4] = a[1];
        yp[5] = a[2];
        yp[6] = -p->derivatives.mass;
    }

    static void vac_state(double t, double y[], double yp[], void* params)
    {
        VacParams* p = reinterpret_cast<VacParams*>(params);
        vac_acceleration(t, y, p);
        Eigen::Vector3d a = p->derivatives.thrust + p->derivatives.gravity;

        yp[0] = y[3];
        yp[1] = y[4];
        yp[2] = y[5];
        yp[3] = a[0];
        yp[4] = a[1];
        yp[5] = a[2];
        yp[6] = -p->derivatives.mass;
    }

    static void output_atm(double t, std::vector<double> y, AtmParams& atm_params, std::vector<double>* output)
    {
        atm_acceleration(t, y.data(), &atm_params);
        double angle = atm_params.initial_angle + t * atm_params.angular_velocity.norm();
        Eigen::Vector3d axis = atm_params.angular_velocity.normalized();
        Eigen::AngleAxisd rotation = Eigen::AngleAxisd(angle, axis);

        output->push_back(atm_params.ndim.time_from(t));
        output->push_back(atm_params.ndim.mass_from(y[6]));
        Eigen::Vector3d r(y.data());
        Eigen::Vector3d v(y.data() + 3);
        std::vector<Eigen::Vector3d> vectors(6);
        vectors[0] = rotation * r;
        vectors[1] = rotation * v + atm_params.angular_velocity.cross(vectors[0]);
        vectors[0] = atm_params.ndim.distance_from(vectors[0]);
        vectors[1] = atm_params.ndim.velocity_from(vectors[1]);
        vectors[2] = atm_params.ndim.acceleration_from(rotation * atm_params.derivatives.drag);
        vectors[3] = atm_params.ndim.acceleration_from(rotation * atm_params.derivatives.thrust);
        vectors[4] = atm_params.ndim.acceleration_from(rotation * atm_params.derivatives.gravity);
        vectors[5] = atm_params.ndim.acceleration_from(rotation * atm_params.derivatives.rotation);

        for (auto& vector : vectors)
        {
            for (auto& value : vector)
            {
                output->push_back(value);
            }
        }
    }

    static void output_vac(double t, std::vector<double> y, VacParams& vac_params, std::vector<double>* output)
    {
        vac_acceleration(t, y.data(), &vac_params);

        output->push_back(vac_params.ndim.time_from(t));
        output->push_back(vac_params.ndim.mass_from(y[6]));
        std::vector<Eigen::Vector3d> vectors(6);
        Eigen::Vector3d r(y.data());
        Eigen::Vector3d v(y.data() + 3);
        vectors[0] = vac_params.ndim.distance_from(r);
        vectors[1] = vac_params.ndim.velocity_from(v);
        vectors[2] = Eigen::Vector3d::Zero();
        vectors[3] = vac_params.ndim.acceleration_from(vac_params.derivatives.thrust);
        vectors[4] = vac_params.ndim.acceleration_from(vac_params.derivatives.gravity);
        vectors[5] = Eigen::Vector3d::Zero();

        for (auto& vector : vectors)
        {
            for (auto& value : vector)
            {
                output->push_back(value);
            }
        }
    }

    static void simulate_atm_phase(double& t, std::vector<double>& y, py::list py_events, AtmParams atm_params, std::vector<double>* output)
    {
        double altitude = 0.0;
        double altitude_old = 0.0;

        auto udpate_altitude = [&]() -> void {
            altitude_old = altitude;
            altitude = std::sqrt(y[0] * y[0] + y[1] * y[1] + y[2] * y[2]);
            };

        atm_params.ndim.state_to(t, y);

        for (size_t i = 0; i < py_events.size(); i++)
        {
            py::tuple py_event = py_events[i];

            atm_params.mass_rate = atm_params.ndim.mass_rate_to(py_event.attr("mdot").cast<double>());
            atm_params.thrust = py_event.attr("spl_thrust").is_none() ? Spl() : py_spl(py_event[6]);

            if (py_event.attr("stage").cast<bool>())
            { 
                y[6] = atm_params.ndim.mass_to(py_event.attr("mf").cast<double>());
            }
   
            double tout = atm_params.ndim.time_to(py_event.attr("tout").cast<double>());

            Equation eq = Equation(atm_state, t, y, "RK32", 1e-9, 1e-9, &atm_params);
            int steps = int(atm_params.ndim.time_from(tout - t)) + 1;
            double dt = (tout - t) / steps;

            for (size_t j = 0; j < steps + 1; j++)
            {
                // if t + j * dt <= pitch_time, int limit = pitch time
                // if between pitch time and pitch time + pitch duration limit = pitch time + pitch duration
                // if t = 0 dont step

                eq.step(t + j * dt);
                eq.get_y(0, 7, y.data());
                std::vector<double> yp = eq.get_yp();
                udpate_altitude();
                if (altitude < altitude_old)
                {
                    throw std::runtime_error("Switch altitude not reached: negative altitude rate.");
                }
                else if (output && altitude < atm_params.switch_altitude)
                {
                    output_atm(eq.get_t(), y, atm_params, output);
                }
                else if (altitude > atm_params.switch_altitude)
                {
                    for (size_t iter = 0; iter < 10; iter++)
                    {
                        double f = altitude - atm_params.switch_altitude;
                        double df = (altitude - altitude_old) / dt;
                        dt = -f / df;
                        t = eq.get_t() + dt;
                        eq.step(t);
                        eq.get_y(0, 7, y.data());
                        udpate_altitude();
                        if (abs(dt) < 1e-8)
                        {
                            if (output)
                            {
                                output_atm(t, y, atm_params, output);
                            }
                            double angle = atm_params.initial_angle + t * atm_params.angular_velocity.norm();
                            Eigen::Vector3d axis = atm_params.angular_velocity.normalized();
                            Eigen::AngleAxisd rotation = Eigen::AngleAxisd(angle, axis);
                            Eigen::Vector3d r(y.data());
                            Eigen::Vector3d v(y.data() + 3);
                            r = rotation * r;
                            v = rotation * v + atm_params.angular_velocity.cross(r);
                            y[0] = r(0);
                            y[1] = r(1);
                            y[2] = r(2);
                            y[3] = v(0);
                            y[4] = v(1);
                            y[5] = v(2);
                            atm_params.ndim.state_from(t, y);
                            return;
                        }
                    }
                    throw std::runtime_error("Switch altitude not reached: refinement iterations exceeded.");
                }
            }
            t = tout;
        }
        throw std::runtime_error("Switch altitude not reached: events completed.");
    }

    static void simulate_vac_phase(double& t, std::vector<double>& y, py::list py_events, VacParams vac_params, std::vector<double>* output)
    {
        double velocity = 0.0;
        double velocity_old = 0.0;

        auto udpate_velocity = [&]() -> void {
            velocity_old = velocity;
            velocity = std::sqrt(y[3] * y[3] + y[4] * y[4] + y[5] * y[5]);
            };

        vac_params.ndim.state_to(t, y);
        vac_params.tg = t;
        double final_time = t + vac_params.final_time;
        bool last = false;

        for (size_t i = 0; i < py_events.size(); i++)
        {
            py::tuple event = py_events[i];
            bool stage = event.attr("stage").cast<bool>();
            double tout = vac_params.ndim.time_to(event.attr("tout").cast<double>());  
            vac_params.mass_rate = vac_params.ndim.mass_rate_to(event.attr("mdot").cast<double>());
            vac_params.thrust = vac_params.ndim.force_to(event.attr("thrust_vac").cast<double>());
            if (tout < t) { continue; }
            if (stage) { 
                double mf = vac_params.ndim.mass_to(event.attr("mf").cast<double>());
                y[6] = mf; 
            }

            Equation eq = Equation(vac_state, t, y, "RK853", 1e-5, 1e-5, &vac_params);

            if (vac_params.final_velocity)
            {
                int steps = int(vac_params.ndim.time_from(tout - t)) + 1;
                double dt = (tout - t) / steps;
                for (size_t j = 0; j < steps + 1; j++)
                {
                    eq.step(t + j * dt);
                    eq.get_y(0, 7, y.data());
                    udpate_velocity();

                    if (velocity > vac_params.final_velocity)
                    {
                        for (size_t iter = 0; iter < 10; iter++)
                        {
                            double f = velocity - vac_params.final_velocity;
                            double df = (velocity - velocity_old) / dt;
                            dt = -f / df;
                            t = eq.get_t() + dt;
                            eq.step(t);
                            eq.get_y(0, 7, y.data());
                            udpate_velocity();
                            if (abs(dt) < 1e-8)
                            {
                                vac_params.ndim.state_from(t, y);
                                return;
                            }    
                        }
                        throw std::runtime_error("Final velocity not reached: refinement iterations exceeded.");
                    }
                }
            }
            else
            {
                if (tout > final_time) 
                {
                    tout = final_time;
                    last = true;
                }

                if (output)
                {
                    int steps = int(vac_params.ndim.time_from(tout - t)) + 1;
                    double dt = (tout - t) / steps;
                    for (size_t j = 0; j < steps + 1; j++)
                    {
                        eq.step(t + j * dt);
                        eq.get_y(0, 7, y.data());
                        output_vac(eq.get_t(), y, vac_params, output);
                    }
                }
                else
                {
                    eq.step(tout);
                    eq.get_y(0, 7, y.data());
                }
                t = tout;
                if (last)
                {
                    vac_params.ndim.state_from(t, y);
                    return;
                }
            }
        }
        throw std::runtime_error("Simulate vacuum phase failed to complete.");
    }

    static AtmParams create_atm_params(py::dict py_p)
    {
        NDim ndim {
           py_p["ndim"]["distance"].cast<double>(),
           py_p["ndim"]["velocity"].cast<double>(),
           py_p["ndim"]["acceleration"].cast<double>(),
           py_p["ndim"]["time"].cast<double>(),
           py_p["ndim"]["mass"].cast<double>()
        };

        return AtmParams{
            ndim,
            DragModel{
                py_spl(py_p["splines"]["pressure"]),
                py_spl(py_p["splines"]["density"]),
                py_spl(py_p["splines"]["drag"]),
                py_spl(py_p["splines"]["drag_mul"])
            },
            Derivatives(),
            py_p["settings"]["initial_angle"].cast<double>(),
            ndim.rate_to(py_vector3d(py_p["body"]["angular_velocity"])),
            ndim.acceleration_to(py_p["settings"]["a_limit"].cast<double>()),
            py_p["settings"]["azimuth"].cast<double>(),
            ndim.time_to(py_p["settings"]["pitch_time"].cast<double>()),
            ndim.time_to(py_p["settings"]["pitch_duration"].cast<double>()),
            ndim.rate_to(py_p["settings"]["pitch_rate"].cast<double>()),
            ndim.distance_to(py_p["settings"]["switch_altitude"].cast<double>()),
            Spl(),
            0.0
        };
    }

    static VacParams create_vac_params(py::dict py_p)
    {
        NDim ndim{
           py_p["ndim"]["distance"].cast<double>(),
           py_p["ndim"]["velocity"].cast<double>(),
           py_p["ndim"]["acceleration"].cast<double>(),
           py_p["ndim"]["time"].cast<double>(),
           py_p["ndim"]["mass"].cast<double>()
        };

        return VacParams{
            ndim,
            Derivatives(),
            ndim.acceleration_to(py_p["settings"]["a_limit"].cast<double>()),
            0.0,
            py_vector3d(py_p["settings"]["x"]),
            py_vector3d(py_p["settings"]["x"], 3),
            py_p["settings"]["x"].cast<py::array_t<double>>().at(6),
            0.0,
            0.0,
            (!py_p["settings"]["vf"].is_none() ? ndim.velocity_to(py_p["settings"]["vf"].cast<double>()) : 0.0)
        };
    }

    // make sure to remove output from this
    py::tuple kgs_simulate_atm_phase(double t, py::array_t<double> py_y, py::list py_events, py::dict py_p)
    {
        AtmParams atm_params = create_atm_params(py_p);

        std::vector<double> y = py_std_vector(py_y);

        //std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        simulate_atm_phase(t, y, py_events, atm_params, nullptr);
        //std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        //std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << '\n';

        return py::make_tuple(t, py_copy_array(y));
    }

    py::tuple kgs_simulate_vac_phase(double t, py::array_t<double> py_y, py::list py_events, py::dict py_p)
    {
        VacParams vac_params = create_vac_params(py_p);

        std::vector<double> y = py_std_vector(py_y);

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        simulate_vac_phase(t, y, py_events, vac_params, nullptr);
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << '\n';

        return py::make_tuple(t, py_copy_array(y));
    }

    py::array_t<double> kgs_constraint_residuals(double t, py::array_t<double> py_y, py::list py_events, py::dict py_p)
    {
        VacParams vac_params = create_vac_params(py_p);

        std::vector<double> y = py_std_vector(py_y);

        simulate_vac_phase(t, y, py_events, vac_params, nullptr);

        py::array_t<double> f({ 1, 7 });
        f = f.reshape({ 7 });
        double* f_ptr = f.mutable_data();

        Eigen::Vector3d rf(y.data());
        Eigen::Vector3d vf(y.data() + 3);
        rf = vac_params.ndim.distance_to(rf);
        vf = vac_params.ndim.velocity_to(vf);

        double tf = vac_params.final_time;

        Eigen::Vector3d lambda_f = vac_params.lambda_i * cos(tf) + vac_params.lambda_dot_i * sin(tf);
        Eigen::Vector3d lambda_dot_f = -vac_params.lambda_i * sin(tf) + vac_params.lambda_dot_i * cos(tf);
        Eigen::Vector3d sigma = rf.cross(lambda_dot_f) - vf.cross(lambda_f);

        py::array_t<double> c = py_p["settings"]["c"].cast<py::array_t<double>>();
        double* c_ptr = c.mutable_data();

        f_ptr[0] = rf.dot(rf) - c_ptr[0];
        f_ptr[1] = vf.dot(vf) - c_ptr[1];
        f_ptr[2] = rf.dot(vf) - c_ptr[2];
        f_ptr[3] = sigma[0];
        f_ptr[4] = sigma[1];
        f_ptr[5] = sigma[2];
        f_ptr[6] = lambda_dot_f.norm() - 1.0;

        return f;
    }

    py::array_t<double> kgs_output_time_series(double t, py::array_t<double> py_y, py::list py_events, py::dict py_p_atm, py::dict py_p_vac)
    {
        std::vector<double> output;

        AtmParams atm_params = create_atm_params(py_p_atm);
        VacParams vac_params = create_vac_params(py_p_vac);

        std::vector<double> y = py_std_vector(py_y);

        simulate_atm_phase(t, y, py_events, atm_params, &output);
        simulate_vac_phase(t, y, py_events, vac_params, &output);

        py::array_t<double> py_output({ int(output.size() / 20) , 20 });
        double* output_ptr = py_output.mutable_data();

        for (int i = 0; i < output.size(); i++)
        {
            output_ptr[i] = output[i];
        }

        return py_output;
    }
}