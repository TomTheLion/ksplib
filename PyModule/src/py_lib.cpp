#include <iostream>
#include <iomanip>
#include <fstream>

#include "nlopt.hpp"

#include "py_lib.h"

#include "Spl.h"
#include "astrodynamics.h"
#include "FlightPlan.h"
#include "LunarFlightPlan.h"
#include "ConicLunarFlightPlan.h"

double* py_arr_ptr(py::object obj)
{
    return static_cast<double*>(obj.cast<py::array_t<double>>().request().ptr);
}

int py_arr_size(py::object obj)
{
    return obj.cast<py::array_t<double>>().request().size;
}

Eigen::Vector3d py_vector3d(py::object obj)
{
    return Eigen::Map<Eigen::Vector3d>(py_arr_ptr(obj.cast<py::array_t<double>>()));
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

template<typename T>
py::list py_copy_list(std::vector<T> x)
{
    py::list py_list;

    for (auto& v : x) py_list.append(v);

    return py_list;
}

template<typename T>
py::array_t<T, py::array::c_style> py_copy_array(std::vector<T> x)
{
    int l = x.size();

    py::array_t<T, py::array::c_style> py_x({ l });
    auto mx = py_x.mutable_unchecked();

    for (int i = 0; i < l; i++)
    {
        mx(i) = x[i];
    }

    return py_x;
}

py::array_t<double, py::array::c_style> py_copy_array(std::vector<std::vector<double>> x)
{
    int l = x.size();
    int m = x[0].size();

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
    int l = x.size();
    int m = x[0].size();
    int n = x[0][0].size();
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

namespace jdate
{
    double to_jdate_from_ktime(double k)
    {
        Jdate jd;
        jd.set_kerbal_time(k);
        return jd.get_julian_date();
    }

    double to_ktime_from_jdate(double j)
    {
        Jdate jd(j);
        return jd.get_kerbal_time();
    }

    double to_jdate_from_date_time(int m, int d, int y, double t)
    {
        Jdate jd(m, d, y, t);
        return jd.get_julian_date();
    }

    py::tuple get_month_day_year(double j)
    {
        Jdate jd(j);
        auto [m, d, y] = jd.get_month_day_year();
        return py::make_tuple(m, d, y);
    }

    double get_time(double j)
    {
        Jdate jd(j);
        return jd.get_time();
    }
}

namespace ephemeris
{
    void build_ephemeris(py::dict py_info, py::list py_bodies)
    {
        try
        {
            std::string output_file = py_info["output_file"].cast<std::string>();;

            int steps = py_info["steps"].cast<int>();
            double dt = py_info["dt"].cast<double>();
            double relerr = py_info["relerr"].cast<double>();
            double abserr = py_info["abserr"].cast<double>();
            double epherr = py_info["epherr"].cast<double>();

            double mu_scale = py_info["mu_scale"].cast<double>();
            double muk = py_info["muk"].cast<double>() / mu_scale;
            double reference_time = py_info["reference_time"].cast<double>();
            double time_scale = py_info["time_scale"].cast<double>();
            double distance_scale = py_info["distance_scale"].cast<double>();
            double velocity_scale = py_info["velocity_scale"].cast<double>();

            std::vector<double> mu;
            int num_bodies = py_bodies.size();
            std::vector<double> yi(6 * num_bodies);
            std::vector<std::string> bodies;

            for (int i = 0; i < num_bodies; i++)
            {
                py::list body_py = py_bodies[i];
                bodies.push_back(body_py[0].cast<std::string>());
                mu.push_back(body_py[1].cast<double>() / mu_scale);
            }

            std::vector<double> info
            {
                reference_time,
                time_scale,
                mu_scale,
                distance_scale,
                velocity_scale,
            };

            for (int i = 0; i < num_bodies; i++)
            {
                py::list body_py = py_bodies[i];
                py::tuple r = body_py[2];
                py::tuple v = body_py[3];

                for (int j = 0; j < 6; j++)
                {
                    if (j < 3)
                    {
                        yi[6 * i + j] = r[j].cast<double>() / distance_scale;
                    }
                    else
                    {
                        yi[6 * i + j] = v[j - 3].cast<double>() / velocity_scale;
                    }
                }
            }

            astrodynamics::NBodyParams params = { num_bodies, muk, mu };

            Equation equation = Equation(astrodynamics::n_body_df, yi.size(), yi.data(), 0.0, relerr, abserr, &params);
            Ephemeris ephemeris_create = Ephemeris(equation, bodies, muk, mu, steps, dt, epherr, info);

            std::cout << " - : Test Initial Ephemeris" << std::endl;
            ephemeris_create.test(equation);
            ephemeris_create.save(output_file);
            Ephemeris ephemeris_load = Ephemeris(output_file);
            std::cout << std::endl << " - : Test Saved Ephemeris" << std::endl;
            ephemeris_load.test(equation);
            std::cout << std::endl;
        }
        catch (std::exception& e)
        {
            std::cerr << "Error during input processing or build:\n";
            std::cerr << e.what() << "\n";
        }
    }


    py::dict load_ephemeris(std::string file_name)
    {
        try
        {
            // attempt to load ephemeris file
            std::ifstream header_in(file_name + ".cfg");
            std::ifstream binary_in(file_name + ".bin", std::ios::in | std::ios::binary);

            if (!header_in || !binary_in)
            {
                throw std::runtime_error("Runtime Error: Ephemeris file not found");
            }

            std::string header_in_line;
            std::vector<std::string> header_in_vector;
            while (std::getline(header_in, header_in_line))
            {
                header_in_vector.push_back(header_in_line.substr(header_in_line.find(": ") + 2));
            }

            py::dict py_ephemeris;

            int size = std::stoi(header_in_vector[0]);

            py_ephemeris["size"] = size;
            py_ephemeris["num_bodies"] = std::stoi(header_in_vector[1]);
            py_ephemeris["time_steps"] = std::stoi(header_in_vector[2]);
            py_ephemeris["delta_time"] = std::stod(header_in_vector[3]);
            py_ephemeris["max_time"] = py_ephemeris["time_steps"] * py_ephemeris["delta_time"];
            py_ephemeris["muk"] = std::stod(header_in_vector[4]);
            py_ephemeris["reference_time"] = std::stod(header_in_vector[5]);
            py_ephemeris["time_scale"] = std::stod(header_in_vector[6]);
            py_ephemeris["mu_scale"] = std::stod(header_in_vector[7]);
            py_ephemeris["distance_scale"] = std::stod(header_in_vector[8]);
            py_ephemeris["velocity_scale"] = std::stod(header_in_vector[9]);

            auto str_to_str_list = [](std::string str, py::list lst)
            {
                std::stringstream ss(str);
                std::string substr;
                while (std::getline(ss, substr, ',')) {
                    substr.erase(remove(substr.begin(), substr.end(), ' '), substr.end());
                    lst.append(substr);
                }
            };

            auto str_to_int_vector = [](std::string str, std::vector<int>& vec)
            {
                vec.clear();
                std::stringstream ss(str);
                std::string substr;
                while (std::getline(ss, substr, ',')) {
                    vec.push_back(std::stoi(substr));
                }
            };

            auto str_to_double_vector = [](std::string str, std::vector<double>& vec)
            {
                vec.clear();
                std::stringstream ss(str);
                std::string substr;
                while (std::getline(ss, substr, ',')) {
                    vec.push_back(std::stod(substr));
                }
            };

            py::list py_bodies;
            std::vector<double> mu;
            std::vector<int> index;
            std::vector<int> num_coef;
            std::vector<double> ephemeris;

            str_to_int_vector(header_in_vector[10], index);
            str_to_int_vector(header_in_vector[11], num_coef);
            str_to_str_list(header_in_vector[12], py_bodies);
            str_to_double_vector(header_in_vector[13], mu);

            ephemeris.resize(size);

            for (int i = 0; i < size; i++)
            {
                binary_in.read((char*)&ephemeris[i], sizeof ephemeris[i]);
            }

            py_ephemeris["bodies"] = py_bodies;
            py_ephemeris["mu"] = py::array_t<double>(mu.size(), mu.data());
            py_ephemeris["index"] = py::array_t<int>(index.size(), index.data());
            py_ephemeris["num_coef"] = py::array_t<int>(num_coef.size(), num_coef.data());
            py_ephemeris["ephemeris"] = py::array_t<double>(ephemeris.size(), ephemeris.data());

            return py_ephemeris;

        }
        catch (std::exception& e)
        {
            std::cerr << "Error during ephemeris load:\n";
            std::cerr << e.what() << "\n";
        }
    }
}

namespace astrodynamics
{
    py::tuple py_kepler(py::array_t<double> py_r0, py::array_t<double> py_v0, double t, double mu, std::optional<double> eps)
    {
        Eigen::Vector3d r0(py_r0.at(0), py_r0.at(1), py_r0.at(2));
        Eigen::Vector3d v0(py_v0.at(0), py_v0.at(1), py_v0.at(2));
        auto [r1, v1] = kepler(r0, v0, t, mu, eps.value());

        py::array_t<double> py_r1(3);
        py::array_t<double> py_v1(3);

        auto mr1 = py_r1.mutable_unchecked();
        mr1(0) = r1(0);
        mr1(1) = r1(1);
        mr1(2) = r1(2);

        auto mv1 = py_v1.mutable_unchecked();
        mv1(0) = v1(0);
        mv1(1) = v1(1);
        mv1(2) = v1(2);

        return py::make_tuple(py_r1, py_v1);
    }

    py::tuple py_lambert(py::array_t<double> py_r0, py::array_t<double> py_r1, double t, double mu, std::optional<double> eps, std::optional<double> d, std::optional<py::array_t<double>> py_n)
    {
        Eigen::Vector3d r0(py_r0.at(0), py_r0.at(1), py_r0.at(2));
        Eigen::Vector3d r1(py_r1.at(0), py_r1.at(1), py_r1.at(2));
        Eigen::Vector3d n;
        if (py_n.has_value())
        {
            n = Eigen::Vector3d(py_n.value().at(0), py_n.value().at(1), py_n.value().at(2));
        }
        else
        {
            n = -Eigen::Vector3d::UnitY();
        }

        auto [v0, v1] = lambert(r0, r1, t, mu, d.value(), n, eps.value());

        py::array_t<double> py_v0(3);
        py::array_t<double> py_v1(3);

        auto mv0 = py_v0.mutable_unchecked();
        mv0(0) = v0(0);
        mv0(1) = v0(1);
        mv0(2) = v0(2);

        auto mv1 = py_v1.mutable_unchecked();
        mv1(0) = v1(0);
        mv1(1) = v1(1);
        mv1(2) = v1(2);

        return py::make_tuple(py_v0, py_v1);
    }

    //double splev(double x, py::object obj)
    //{
    //    Spl s = py_spl(obj);
    //    return s.eval(x);
    //}

    //double bisplev(double x, double y, py::object obj)
    //{
    //    Spl s = py_bspl(obj);
    //    return s.eval(x, y);
    //}
}

namespace interplanetary
{
    py::dict py_output_result(FlightPlan::Result result)
    {
        py::dict py_result;

        py_result["nlopt_code"] = result.nlopt_code;
        py_result["nlopt_num_evals"] = result.nlopt_num_evals;
        py_result["nlopt_value"] = result.nlopt_value;
        py_result["nlopt_solution"] = py_copy_array(result.nlopt_solution);
        py_result["nlopt_constraints"] = py_copy_array(result.nlopt_constraints);
        py_result["time_scale"] = result.time_scale;
        py_result["distance_scale"] = result.distance_scale;
        py_result["velocity_scale"] = result.velocity_scale;
        py_result["sequence"] = py_copy_array(result.sequence);
        py_result["bodies"] = py_copy_list(result.bodies);
        py_result["mu"] = py::array_t<double>(result.mu.size(), result.mu.data());
        py_result["julian_time"] = py::array_t<double>(result.julian_time.size(), result.julian_time.data());
        py_result["kerbal_time"] = py::array_t<double>(result.kerbal_time.size(), result.kerbal_time.data());
        py_result["phase"] = py::array_t<int>(result.phase.size(), result.phase.data());
        py_result["leg"] = py::array_t<int>(result.leg.size(), result.leg.data());
        py_result["body"] = py::array_t<int>(result.body.size(), result.body.data());
        py_result["r"] = py_copy_array(result.r);
        py_result["v"] = py_copy_array(result.v);
        py_result["rb"] = py_copy_array(result.rb);
        py_result["vb"] = py_copy_array(result.vb);

        return py_result;
    }

    py::dict interplanetary(py::dict py_p, int mode, bool show_errors)
    {
        try
        {
            Ephemeris ephemeris = Ephemeris(py_p["ephemeris"].cast<std::string>());
            FlightPlan flight_plan(ephemeris);

            for (auto& py_flyby : py_p["flybys"].cast<py::list>())
            {
                flight_plan.add_body(
                    py_flyby["body"].cast<std::string>(),
                    Jdate(py_flyby["date"].cast<double>()),
                    py_flyby["radius"].cast<double>(),
                    py_flyby["distance_scale"].cast<double>(),
                    py_flyby["orbits"].cast<double>()
                );
            }

            if (!py_p["min_start_time"].is_none()) flight_plan.add_min_start_time_constraint(py_p["min_start_time"].cast<double>());
            if (!py_p["min_time"].is_none()) flight_plan.add_min_flight_time_constraint(py_p["min_time"].cast<double>());
            if (!py_p["max_time"].is_none()) flight_plan.add_max_flight_time_constraint(py_p["max_time"].cast<double>());
            if (!py_p["max_c3"].is_none()) flight_plan.add_max_c3_constraint(py_p["max_c3"].cast<double>());
            if (!py_p["min_inclination_launch"].is_none() && !py_p["max_inclination_launch"].is_none() && !py_p["n_launch"].is_none())
            {
                py::array_t<double> py_n_launch = py_p["n_launch"];
                Eigen::Vector3d n_launch = { py_n_launch.at(0), py_n_launch.at(1), py_n_launch.at(2) };
                flight_plan.add_inclination_constraint(
                    true, py_p["min_inclination_launch"].cast<double>(), py_p["max_inclination_launch"].cast<double>(), n_launch);
            }

            if (!py_p["min_inclination_arrival"].is_none() && !py_p["max_inclination_arrival"].is_none() && !py_p["n_arrival"].is_none())
            {
                py::array_t<double> py_n_arrival = py_p["n_arrival"];
                Eigen::Vector3d n_arrival = { py_n_arrival.at(0), py_n_arrival.at(1), py_n_arrival.at(2) };
                flight_plan.add_inclination_constraint(
                    false, py_p["min_inclination_arrival"].cast<double>(), py_p["max_inclination_arrival"].cast<double>(), n_arrival);
            }

            if (!py_p["eccentricity_arrival"].is_none()) flight_plan.add_arrival_eccentricity_constraint(py_p["eccentricity_arrival"].cast<double>());

            flight_plan.init_conic_model(py_p["eps_conic"].cast<double>());

            if (!py_p["conic_solution"].is_none())
            {
                py::array_t<double> py_xa = py_p["conic_solution"];
                std::vector<double> xa;
                for (int i = 0; i < py_xa.size(); i++)
                {
                    xa.push_back(py_xa.at(i));
                }
                flight_plan.set_conic_solution(xa);
            }

            if (mode > 0) flight_plan.run_conic_model(py_p["num_evals_conic"].cast<int>(), py_p["eps_conic"].cast<double>(), py_p["eps_conic_t"].cast<double>(), py_p["eps_conic_x"].cast<double>());

            FlightPlan::Result conic_result = flight_plan.output_conic_result(py_p["eps_conic"].cast<double>());

            py::dict output;
            output["conic"] = py_output_result(conic_result);

            if (mode > 1)
            {
                flight_plan.init_nbody_model(py_p["eps_nbody"].cast<double>());

                if (!py_p["nbody_solution"].is_none())
                {
                    py::array_t<double> py_xn = py_p["nbody_solution"];
                    std::vector<double> xn;
                    for (int i = 0; i < py_xn.size(); i++)
                    {
                        xn.push_back(py_xn.at(i));
                    }
                    flight_plan.set_nbody_solution(xn);
                }

                if (mode > 2) flight_plan.run_nbody_model(py_p["num_evals_nbody"].cast<int>(), py_p["eps_nbody"].cast<double>(), py_p["eps_nbody_t"].cast<double>(), py_p["eps_nbody_f"].cast<double>());
                if (mode > 3)
                {
                    if (!py_p["start_time"].is_none()) flight_plan.add_start_time_constraint(py_p["start_time"].cast<double>());
                    flight_plan.run_nbody_model(py_p["num_evals_nbody"].cast<int>(), py_p["eps_nbody"].cast<double>(), py_p["eps_nbody_t"].cast<double>(), py_p["eps_nbody_f"].cast<double>());
                }

                FlightPlan::Result nbody_result = flight_plan.output_nbody_result(py_p["eps_nbody"].cast<double>());
                output["nbody"] = py_output_result(nbody_result);
            }

            return output;
        }
        catch (const std::exception& e)
        {
            if (show_errors)
            {
                std::cerr << e.what() << '\n';
            }      
            return py::dict();
        }
    }

    struct TCMData
    {
        int bref;
        std::vector<double> t;
        std::vector<double> yi;
        astrodynamics::NBodyParams n_body_params;
        std::vector<double> xf;
    };

    void constraints_tcm(unsigned m, double* result, unsigned n, const double* x, double* grad, void* f_data)
    {
        TCMData* data = reinterpret_cast<TCMData*>(f_data);

        std::vector<Eigen::Matrix<double, 6, 6>> stms(data->t.size());

        Equation equation = Equation(astrodynamics::n_body_df_vessel, data->yi.size(), data->yi.data(), data->t[0], 1e-12, 1e-12, &data->n_body_params);

        for (int i = 0; i < data->t.size() - 1; i++)
        {
            double t0 = data->t[i];
            double t1 = data->t[i + 1];
            equation.step(t1);
            equation.get_y(6, 36, stms[i].data());
            equation.set_ti(t1);
            equation.set_yi(0, equation.get_y(0));
            equation.set_yi(1, equation.get_y(1));
            equation.set_yi(2, equation.get_y(2));
            equation.set_yi(3, equation.get_y(3) + x[3 * i + 0]);
            equation.set_yi(4, equation.get_y(4) + x[3 * i + 1]);
            equation.set_yi(5, equation.get_y(5) + x[3 * i + 2]);

            for (int j = 42; j < equation.get_neqn(); j++)
            {
                equation.set_yi(j, equation.get_y(j));
            }
            equation.reset();
        }
        equation.get_y(6, 36, stms.back().data());

        for (int i = 0; i < 6; i++)
        {    
            result[i] = equation.get_y(i) - data->xf[i];
            if (data->bref > -1)
            {
                result[i] -= equation.get_y(42 + 6 * data->bref + i);
            }
        }

        if (grad)
        {
            Eigen::MatrixXd mgrad(m, n);
            for (int i = 1; i < data->t.size(); i++)
            {
                Eigen::Matrix<double, 6, 6> stm = stms[i];
                for (int j = i + 1; j < stms.size(); j++)
                {
                    stm = stms[j] * stm;
                }
                mgrad.block<6, 3>(0, 3 * (i - 1)) = stm.block<6, 3>(0, 3);
            }

            for (int j = 0; j < m; j++)
            {
                for (int i = 0; i < n; i++)
                {
                    grad[j * n + i] = mgrad(j, i);
                }
            }
        }
    }

    double objective_tcm(unsigned n, const double* x, double* grad, void* f_data)
    {
        TCMData* data = reinterpret_cast<TCMData*>(f_data);

        int b = 0;
        if (data->bref == -1)
        {
            b = 1;
        }

        double f = 0.0;
        for (int i = 0; i < n / 3 - b; i++)
        {
            f += sqrt(
                x[3 * i + 0] * x[3 * i + 0] +
                x[3 * i + 1] * x[3 * i + 1] +
                x[3 * i + 2] * x[3 * i + 2]
            );
        }

        if (grad)
        {
            for (int i = 0; i < n / 3; i++)
            {
                double dv = sqrt(
                    x[3 * i + 0] * x[3 * i + 0] +
                    x[3 * i + 1] * x[3 * i + 1] +
                    x[3 * i + 2] * x[3 * i + 2]
                );

                grad[3 * i + 0] = x[3 * i + 0] / dv;
                grad[3 * i + 1] = x[3 * i + 1] / dv;
                grad[3 * i + 2] = x[3 * i + 2] / dv;
            }
        }

        return f;
    }

    py::tuple trajectory_correction(py::array_t<double> py_r, py::array_t<double> py_v, py::array_t<double> py_rt, py::array_t<double> py_vt, py::array_t<double> py_t, int bref, py::tuple py_body_info)
    {
        try
        {
            py::list py_bodies = py_body_info[1];
            int num_bodies = py_bodies.size();
            std::vector<double> mu;

            for (int i = 0; i < num_bodies; i++)
            {
                py::list py_body = py_bodies[i];
                mu.push_back(py_body[1].cast<double>());
            }

            TCMData tcm_data;

            tcm_data.bref = bref;
            tcm_data.n_body_params.num_bodies = num_bodies;
            tcm_data.n_body_params.muk = py_body_info[0].cast<double>();
            tcm_data.n_body_params.mu = mu;
            tcm_data.yi.resize(42 + 6 * num_bodies, 0.0);

            for (int i = 0; i < py_t.size(); i++)
            {
                tcm_data.t.push_back(py_t.at(i));
            }

            for (int i = 0; i < 3; i++)
            {
                tcm_data.xf.push_back(py_rt.at(i));
            }
            for (int i = 0; i < 3; i++)
            {
                tcm_data.xf.push_back(py_vt.at(i));
            }

            for (int i = 0; i < 3; i++)
            {
                tcm_data.yi[i] = py_r.at(i);
                tcm_data.yi[i + 3] = py_v.at(i);
            }

            std::vector<double> stmi = {
                1, 0, 0, 0, 0, 0,
                0, 1, 0, 0, 0, 0,
                0, 0, 1, 0, 0, 0,
                0, 0, 0, 1, 0, 0,
                0, 0, 0, 0, 1, 0,
                0, 0, 0, 0, 0, 1
            };

            for (int i = 0; i < 36; i++)
            {
                tcm_data.yi[6 + i] = stmi[i];
            }

            for (int i = 0; i < num_bodies; i++)
            {
                py::list py_body = py_bodies[i];
                py::tuple r = py_body[2];
                py::tuple v = py_body[3];

                for (int j = 0; j < 6; j++)
                {
                    if (j < 3)
                    {
                        tcm_data.yi[42 + 6 * i + j] = r[j].cast<double>();
                    }
                    else
                    {
                        tcm_data.yi[42 + 6 * i + j] = v[j - 3].cast<double>();
                    }
                }
            }

            int m = 6;
            int n = (tcm_data.t.size() - 1) * 3;
            std::vector<double> x0(n, 1e-3);

            std::vector<double> tol{ 5.7735e2, 5.7735e2, 5.7735e2, 5.7735e-4, 5.7735e-4, 5.7735e-4 };
            double minf;

            nlopt::opt opt = nlopt::opt("LD_SLSQP", n);
            opt.set_min_objective(objective_tcm, &tcm_data);
            opt.add_equality_mconstraint(constraints_tcm, &tcm_data, tol);
            opt.set_ftol_abs(1e-4);
            opt.optimize(x0, minf);

            std::cout << std::setprecision(17);
            std::cout << "last optimum value: " << opt.last_optimum_value() << '\n';
            std::cout << "last optimum result: " << opt.last_optimize_result() << '\n';
            std::cout << "num evals: " << opt.get_numevals() << '\n';

            std::vector<double> result(6);

            constraints_tcm(m, result.data(), n, x0.data(), NULL, &tcm_data);

            py::list py_list_x0;

            for (int i = 0; i < x0.size() / 3; i++)
            {
                py::array_t<double> dv(3);
                auto mdv = dv.mutable_unchecked();
                mdv(0) = x0[3 * i + 0];
                mdv(1) = x0[3 * i + 1];
                mdv(2) = x0[3 * i + 2];
                py_list_x0.append(dv);
            }

            return py_list_x0;
        }
        catch (std::exception& e)
        {
            std::cerr << e.what() << "\n";
        }
    }
}

namespace lunar
{
    py::dict py_output_result(LunarFlightPlan::Result result)
    {
        py::dict py_result;

        py_result["nlopt_code"] = result.nlopt_code;
        py_result["nlopt_num_evals"] = result.nlopt_num_evals;
        py_result["nlopt_value"] = result.nlopt_value;
        py_result["nlopt_solution"] = py_copy_array(result.nlopt_solution);
        py_result["nlopt_constraints"] = py_copy_array(result.nlopt_constraints);
        py_result["time_scale"] = result.time_scale;
        py_result["distance_scale"] = result.distance_scale;
        py_result["velocity_scale"] = result.velocity_scale;
        py_result["bodies"] = py_copy_list(result.bodies);
        py_result["muk"] = result.muk;
        py_result["mu"] = py::array_t<double>(result.mu.size(), result.mu.data());
        py_result["julian_time"] = py::array_t<double>(result.julian_time.size(), result.julian_time.data());
        py_result["kerbal_time"] = py::array_t<double>(result.kerbal_time.size(), result.kerbal_time.data());
        py_result["leg"] = py::array_t<int>(result.leg.size(), result.leg.data());
        py_result["r"] = py_copy_array(result.r);
        py_result["v"] = py_copy_array(result.v);
        py_result["rsun"] = py_copy_array(result.rsun);
        py_result["vsun"] = py_copy_array(result.vsun);
        py_result["rmoon"] = py_copy_array(result.rmoon);
        py_result["vmoon"] = py_copy_array(result.vmoon);

        return py_result;
    }

    py::dict lunar(py::dict py_p)
    {
        try
        {
            Ephemeris ephemeris = Ephemeris(py_p["ephemeris"].cast<std::string>());
            LunarFlightPlan flight_plan(ephemeris);


            std::string py_mode = py_p["mode"].cast<std::string>();
            
            LunarFlightPlan::TrajectoryMode mode;

            if (py_mode == "free_return")
            {
                mode = LunarFlightPlan::TrajectoryMode::FREE_RETURN;
            }
            else if (py_mode == "leave")
            {
                mode = LunarFlightPlan::TrajectoryMode::LEAVE;
            }
            else if (py_mode == "return")
            {
                mode = LunarFlightPlan::TrajectoryMode::RETURN;
            }

            Jdate initial_time(py_p["initial_time"].cast<double>());
            double rp_earth = py_p["rp_earth"].cast<double>();
            double rp_moon = py_p["rp_moon"].cast<double>();
            double e_moon = py_p["e_moon"].cast<double>();

            flight_plan.set_mission(initial_time, mode, rp_earth, rp_moon, e_moon);

            if (!py_p["min_time"].is_none()) flight_plan.add_min_flight_time_constraint(py_p["min_time"].cast<double>());
            if (!py_p["max_time"].is_none()) flight_plan.add_max_flight_time_constraint(py_p["max_time"].cast<double>());
            if (!py_p["min_inclination_launch"].is_none() && !py_p["max_inclination_launch"].is_none() && !py_p["n_launch"].is_none())
            {
                py::array_t<double> py_n_launch = py_p["n_launch"];
                Eigen::Vector3d n_launch = { py_n_launch.at(0), py_n_launch.at(1), py_n_launch.at(2) };
                flight_plan.add_inclination_constraint(
                    true, py_p["min_inclination_launch"].cast<double>(), py_p["max_inclination_launch"].cast<double>(), n_launch);
            }

            if (!py_p["min_inclination_arrival"].is_none() && !py_p["max_inclination_arrival"].is_none() && !py_p["n_arrival"].is_none())
            {
                py::array_t<double> py_n_arrival = py_p["n_arrival"];
                Eigen::Vector3d n_arrival = { py_n_arrival.at(0), py_n_arrival.at(1), py_n_arrival.at(2) };
                flight_plan.add_inclination_constraint(
                    false, py_p["min_inclination_arrival"].cast<double>(), py_p["max_inclination_arrival"].cast<double>(), n_arrival);
            }

            flight_plan.init_model();

            flight_plan.run_model(py_p["num_evals"].cast<int>(), py_p["eps"].cast<double>(), py_p["eps_t"].cast<double>(), py_p["eps_x"].cast<double>());

            LunarFlightPlan::Result result = flight_plan.output_result(py_p["eps"].cast<double>());

            return py_output_result(result);
        }
        catch (const std::exception& e)
        {
            std::cerr << e.what() << '\n';
            return py::dict();
        }
    }
}

namespace kerbal_guidance_system
{
    // python interface functions
    const double pi = 3.14159265358979323846;

    // enums, structs, and init functions
    enum class StageType
    {
        ConstThrust,
        ConstAccel,
        Coast,
        NextStage,
        Jettison
    };

    struct AtmStateParams
    {
        double mu;
        double pitch_time;
        double pitch_angle;
        double azimuth;
        double a_limit;
        double thrust_vac;
        double isp_slv;
        double isp_vac;
        double mass_rate;
        Spl pressure;
        Spl density;
        Spl cds;
    };

    AtmStateParams init_atm_state_params(py::dict py_p)
    {

        AtmStateParams p{
            py_p["mu"].cast<double>(),
            py_p["pitch_time"].cast<double>(),
            py_p["pitch_angle"].cast<double>(),
            py_p["azimuth"].cast<double>(),
            py_p["a_limit"].cast<double>(),
            py_p["thrust_vac"].cast<double>(),
            py_p["isp_slv"].cast<double>(),
            py_p["isp_vac"].cast<double>(),
            py_p["mass_rate"].cast<double>(),
            py_spl(py_p["spl_pressure"]),
            py_spl(py_p["spl_density"]),
            py_spl(py_p["spl_cds"])
        };

        return p;
    }

    struct VacStateParams
    {
        StageType mode;
        double tg;
        double a_limit;
        double thrust;
        double exhaust_velocity;
        Eigen::Vector3d lambda_i;
        Eigen::Vector3d lambda_dot_i;
    };

    VacStateParams init_vac_state_params(py::dict py_p)
    {

        VacStateParams p{
            static_cast<StageType>(py_p["mode"].cast<int>()),
            py_p["tg"].cast<double>(),
            py_p["a_limit"].cast<double>(),
            py_p["thrust"].cast<double>(),
            py_p["exhaust_velocity"].cast<double>(),
            py_vector3d(py_p["lambda_i"]),
            py_vector3d(py_p["lambda_dot_i"])
        };

        return p;
    }

    struct OrbitBurnParams
    {
        double mu;
        double thrust;
        double exhaust_velocity;
        Eigen::Vector3d attitude;
        Eigen::Vector3d attitude_rate;
    };

    OrbitBurnParams init_orbit_burn_params(py::dict py_p)
    {

        OrbitBurnParams p{
            py_p["mu"].cast<double>(),
            py_p["thrust"].cast<double>(),
            py_p["exhaust_velocity"].cast<double>(),
            py_vector3d(py_p["attitude"]),
            py_vector3d(py_p["attitude_rate"])
        };

        return p;
    }

    // derivative functions
    void atm_state(double t, double y[], double yp[], void* params)
    {

        AtmStateParams* p = reinterpret_cast<AtmStateParams*>(params);

        Eigen::Vector3d r = { y[0], y[1], y[2] };
        Eigen::Vector3d v = { y[3], y[4], y[5] };
        double m = y[6];

        double pitch = pi / 2.0;
        if (t > p->pitch_time) {
            pitch -= astrodynamics::vector_angle(r, v);
            pitch -= p->pitch_angle * pow(5729.57795130823 * p->pitch_angle, -0.01 * pow(t - p->pitch_time - 10.0, 2));
        }

        double throttle = p->a_limit ? std::min(1.0, p->a_limit / (p->thrust_vac / m)) : 1.0;

        Eigen::Vector3d attitude = Eigen::AngleAxisd(p->azimuth, r.normalized()) * r.cross(Eigen::Vector3d::UnitY()).normalized();
        attitude = Eigen::AngleAxisd(pitch, attitude.cross(r).normalized()) * attitude;

        double pressure = exp(p->pressure.eval(r.norm()));
        double density = exp(p->density.eval(r.norm()));
        double mach = v.norm() / sqrt(1.4 * pressure / density);
        double isp = p->isp_vac - (p->isp_vac - p->isp_slv) * pressure / 101325.0;
        double cds = p->cds.eval(mach);

        Eigen::Vector3d a_thrust = p->thrust_vac * isp / p->isp_vac / m * attitude * throttle;
        Eigen::Vector3d a_drag = -v.normalized() * cds * density * v.squaredNorm() / m;
        Eigen::Vector3d a_gravity = -p->mu * r / pow(r.norm(), 3);

        Eigen::Vector3d a = a_thrust + a_drag + a_gravity;

        yp[0] = v[0];
        yp[1] = v[1];
        yp[2] = v[2];
        yp[3] = a[0];
        yp[4] = a[1];
        yp[5] = a[2];
        yp[6] = -p->mass_rate * throttle;
    }

    void vac_state(double t, double y[], double yp[], void* params)
    {

        VacStateParams* p = reinterpret_cast<VacStateParams*>(params);

        Eigen::Vector3d r = { y[0], y[1], y[2] };
        Eigen::Vector3d v = { y[3], y[4], y[5] };
        double m = y[6];

        double a_thrust_mag = 0.0;
        if (p->mode == StageType::ConstThrust) {
            a_thrust_mag = p->thrust / m;
        }
        else if (p->mode == StageType::ConstAccel) {
            a_thrust_mag = p->a_limit;
        }

        Eigen::Vector3d lambda = p->lambda_i * cos(t - p->tg) + p->lambda_dot_i * sin(t - p->tg);
        Eigen::Vector3d a = -r / pow(r.norm(), 3) + a_thrust_mag * lambda.normalized();

        yp[0] = v[0];
        yp[1] = v[1];
        yp[2] = v[2];
        yp[3] = a[0];
        yp[4] = a[1];
        yp[5] = a[2];
        yp[6] = -a_thrust_mag * m / p->exhaust_velocity;
    }

    void output_atm_state(double t, double y[], double output[], void* params)
    {

        AtmStateParams* p = reinterpret_cast<AtmStateParams*>(params);

        Eigen::Vector3d r = { y[0], y[1], y[2] };
        Eigen::Vector3d v = { y[3], y[4], y[5] };
        double m = y[6];

        double pitch = pi / 2.0;
        if (t > p->pitch_time) {
            pitch -= astrodynamics::vector_angle(r, v);
            pitch -= p->pitch_angle * pow(5729.57795130823 * p->pitch_angle, -0.01 * pow(t - p->pitch_time - 10.0, 2));
        }

        double throttle = p->a_limit ? std::min(1.0, p->a_limit / (p->thrust_vac / m)) : 1.0;

        Eigen::Vector3d attitude = Eigen::AngleAxisd(p->azimuth, r.normalized()) * r.cross(Eigen::Vector3d::UnitY()).normalized();
        attitude = Eigen::AngleAxisd(pitch, attitude.cross(r).normalized()) * attitude;

        double pressure = exp(p->pressure.eval(r.norm()));
        double density = exp(p->density.eval(r.norm()));
        double mach = v.norm() / sqrt(1.4 * pressure / density);
        double isp = p->isp_vac - (p->isp_vac - p->isp_slv) * pressure / 101325.0;
        double cds = p->cds.eval(mach);

        Eigen::Vector3d a_thrust = p->thrust_vac * isp / p->isp_vac / m * attitude * throttle;
        Eigen::Vector3d a_drag = -v.normalized() * cds * density * v.squaredNorm() / m;
        Eigen::Vector3d a_gravity = -p->mu * r / pow(r.norm(), 3);

        output[0] = t;
        for (int i = 0; i < 3; i++) {
            output[i + 1] = r[i];
            output[i + 4] = v[i];
            output[i + 7] = a_thrust[i];
            output[i + 10] = a_drag[i];
            output[i + 13] = a_gravity[i];
        }
    }

    void output_vac_state(double t, double y[], double output[], void* params)
    {

        VacStateParams* p = reinterpret_cast<VacStateParams*>(params);

        Eigen::Vector3d r = { y[0], y[1], y[2] };
        Eigen::Vector3d v = { y[3], y[4], y[5] };
        double m = y[6];

        double a_thrust_mag = 0.0;
        if (p->mode == StageType::ConstThrust) {
            a_thrust_mag = p->thrust / m;
        }
        else if (p->mode == StageType::ConstAccel) {
            a_thrust_mag = p->a_limit;
        }

        Eigen::Vector3d lambda = p->lambda_i * cos(t - p->tg) + p->lambda_dot_i * sin(t - p->tg);
        Eigen::Vector3d a_gravity = -r / pow(r.norm(), 3);
        Eigen::Vector3d a_thrust = a_thrust_mag * lambda.normalized();

        output[0] = t;
        for (int i = 0; i < 3; i++) {
            output[i + 1] = r[i];
            output[i + 4] = v[i];
            output[i + 7] = a_thrust[i];
            output[i + 10] = 0.0;
            output[i + 13] = a_gravity[i];
        }
    }

    void orbit_burn_state(double t, double y[], double yp[], void* params)
    {

        OrbitBurnParams* p = reinterpret_cast<OrbitBurnParams*>(params);

        Eigen::Vector3d r = { y[0], y[1], y[2] };
        Eigen::Vector3d v = { y[3], y[4], y[5] };
        double m = y[6];

        double at = p->thrust / m;
        Eigen::Vector3d lambda = p->attitude + p->attitude_rate * t;
        Eigen::Vector3d a = -r * p->mu / pow(r.norm(), 3) + lambda.normalized() * at;

        yp[0] = v[0];
        yp[1] = v[1];
        yp[2] = v[2];
        yp[3] = a[0];
        yp[4] = a[1];
        yp[5] = a[2];
        yp[6] = -at * m / p->exhaust_velocity;
    }

    // solver functions
    void solve_ivp(double t, double tout, py::array_t<double> py_y, py::dict py_p, void f(double t, double y[], double yp[], void* params), void* params)
    {
        double* y = py_arr_ptr(py_y);
        double relerr = py_p["relerr"].cast<double>();
        double abserr = py_p["relerr"].cast<double>();
        Equation eq = Equation(f, 7, y, t, relerr, abserr, params);
        eq.step(tout);
        eq.get_y(0, 7, y);
    }

    void solve_atm_ivp(double t, double tout, py::array_t<double> py_y, py::dict py_p)
    {
        AtmStateParams p = init_atm_state_params(py_p);
        solve_ivp(t, tout, py_y, py_p, atm_state, &p);
    }

    void solve_vac_ivp(double t, double tout, py::object py_y, py::dict py_p)
    {
        VacStateParams p = init_vac_state_params(py_p);
        solve_ivp(t, tout, py_y, py_p, vac_state, &p);
    }

    void solve_orbit_burn(double t, double tout, py::object py_y, py::dict py_p)
    {
        OrbitBurnParams p = init_orbit_burn_params(py_p);
        solve_ivp(t, tout, py_y, py_p, orbit_burn_state, &p);
    }

    // output functions
    py::array_t<double> output_ivp(double t, double tout, int steps, py::object py_y, py::dict py_p, void f(double t, double y[], double yp[], void* params), void f_output(double t, double y[], double yp[], void* params), void* params)
    {
        double* y = py_arr_ptr(py_y);
        double relerr = py_p["relerr"].cast<double>();
        double abserr = py_p["relerr"].cast<double>();
        Equation eq = Equation(f, 7, y, t, relerr, abserr, params);
        py::array_t<double> output({ steps, 16 });
        double* output_ptr = py_arr_ptr(output);
        double dt = (tout - t) / steps;
        for (int i = 0; i < steps; i++) {
            double tout = t + (i + 1) * dt;
            eq.step(tout);
            eq.get_y(0, 7, y);
            f_output(tout, y, &output_ptr[16 * i], params);
        }

        return output;
    }

    py::array_t<double> output_atm_ivp(double t, double tout, int steps, py::object py_y, py::dict py_p)
    {
        AtmStateParams p = init_atm_state_params(py_p);
        return output_ivp(t, tout, steps, py_y, py_p, atm_state, output_atm_state, &p);
    }

    py::array_t<double> output_vac_ivp(double t, double tout, int steps, py::object py_y, py::dict py_p)
    {
        VacStateParams p = init_vac_state_params(py_p);
        return output_ivp(t, tout, steps, py_y, py_p, vac_state, output_vac_state, &p);
    }
}

namespace conic
{
    py::dict py_output_result(ConicLunarFlightPlan::Result result)
    {
        py::dict py_result;

        py_result["nlopt_code"] = result.nlopt_code;
        py_result["nlopt_num_evals"] = result.nlopt_num_evals;
        py_result["nlopt_value"] = result.nlopt_value;
        py_result["nlopt_solution"] = py_copy_array(result.nlopt_solution);
        py_result["nlopt_constraints"] = py_copy_array(result.nlopt_constraints);
        py_result["time"] = py::array_t<double>(result.time.size(), result.time.data());
        py_result["leg"] = py::array_t<int>(result.leg.size(), result.leg.data());
        py_result["r"] = py_copy_array(result.r);
        py_result["v"] = py_copy_array(result.v);
        py_result["rmoon"] = py_copy_array(result.rmoon);
        py_result["vmoon"] = py_copy_array(result.vmoon);

        return py_result;
    }

    py::dict lunar(py::dict py_p)
    {
        try
        {
            Orbit planet_orbit = Orbit(
                py_p["planet"]["orbit"]["gravitational_parameter"].cast<double>(),		// gravitational_parameter
                py_p["planet"]["orbit"]["semi_major_axis"].cast<double>(),         	    // semi_major_axis
                py_p["planet"]["orbit"]["eccentricity"].cast<double>(),				    // eccentricity
                py_p["planet"]["orbit"]["inclination"].cast<double>(),				    // inclination
                py_p["planet"]["orbit"]["longitude_of_ascending_node"].cast<double>(),	// longitude_of_ascending_node
                py_p["planet"]["orbit"]["argument_of_periapsis"].cast<double>(),		// argument_of_periapsis
                py_p["planet"]["orbit"]["mean_anomaly_at_epoch"].cast<double>(),		// mean_anomaly_at_epoch
                py_p["planet"]["orbit"]["epoch"].cast<double>()                         // epoch
            );				       

            Orbit moon_orbit = Orbit(
                py_p["moon"]["orbit"]["gravitational_parameter"].cast<double>(),		// gravitational_parameter
                py_p["moon"]["orbit"]["semi_major_axis"].cast<double>(),         	    // semi_major_axis
                py_p["moon"]["orbit"]["eccentricity"].cast<double>(),				    // eccentricity
                py_p["moon"]["orbit"]["inclination"].cast<double>(),				    // inclination
                py_p["moon"]["orbit"]["longitude_of_ascending_node"].cast<double>(),	// longitude_of_ascending_node
                py_p["moon"]["orbit"]["argument_of_periapsis"].cast<double>(),		    // argument_of_periapsis
                py_p["moon"]["orbit"]["mean_anomaly_at_epoch"].cast<double>(),		    // mean_anomaly_at_epoch
                py_p["moon"]["orbit"]["epoch"].cast<double>()                           // epoch
            );

            astrodynamics::ConicBody moon = { &moon_orbit, py_p["moon"]["gravitational_parameter"].cast<double>(), py_p["moon"]["soi"].cast<double>(), py_p["moon"]["radius"].cast<double>() };
            astrodynamics::ConicBody planet = { &planet_orbit, py_p["planet"]["gravitational_parameter"].cast<double>(), py_p["planet"]["soi"].cast<double>(), py_p["planet"]["radius"].cast<double>() };

            ConicLunarFlightPlan flight_plan(planet, moon);

            std::string py_mode = py_p["mode"].cast<std::string>();

            ConicLunarFlightPlan::TrajectoryMode mode;

            if (py_mode == "free_return")
            {
                mode = ConicLunarFlightPlan::TrajectoryMode::FREE_RETURN;
            }
            else if (py_mode == "leave")
            {
                mode = ConicLunarFlightPlan::TrajectoryMode::LEAVE;
            }
            else if (py_mode == "return")
            {
                mode = ConicLunarFlightPlan::TrajectoryMode::RETURN;
            }

            double initial_time(py_p["initial_time"].cast<double>());
            double rp_planet = py_p["rp_planet"].cast<double>();
            double rp_moon = py_p["rp_moon"].cast<double>();
            double e_moon = py_p["e_moon"].cast<double>();

            flight_plan.set_mission(initial_time, mode, rp_planet, rp_moon, e_moon);

            if (!py_p["min_time"].is_none()) flight_plan.add_min_flight_time_constraint(py_p["min_time"].cast<double>());
            if (!py_p["max_time"].is_none()) flight_plan.add_max_flight_time_constraint(py_p["max_time"].cast<double>());
            if (!py_p["min_inclination_launch"].is_none() && !py_p["max_inclination_launch"].is_none() && !py_p["n_launch"].is_none())
            {
                py::array_t<double> py_n_launch = py_p["n_launch"];
                Eigen::Vector3d n_launch = { py_n_launch.at(0), py_n_launch.at(1), py_n_launch.at(2) };
                flight_plan.add_inclination_constraint(
                    true, py_p["min_inclination_launch"].cast<double>(), py_p["max_inclination_launch"].cast<double>(), n_launch);
            }

            if (!py_p["min_inclination_arrival"].is_none() && !py_p["max_inclination_arrival"].is_none() && !py_p["n_arrival"].is_none())
            {
                py::array_t<double> py_n_arrival = py_p["n_arrival"];
                Eigen::Vector3d n_arrival = { py_n_arrival.at(0), py_n_arrival.at(1), py_n_arrival.at(2) };
                flight_plan.add_inclination_constraint(
                    false, py_p["min_inclination_arrival"].cast<double>(), py_p["max_inclination_arrival"].cast<double>(), n_arrival);
            }

            if (!py_p["n_launch_direction"].is_none() && !py_p["n_launch_plane"].is_none())
            {
                py::array_t<double> py_n_launch_plane = py_p["n_launch_plane"];
                Eigen::Vector3d n_launch_plane = { py_n_launch_plane.at(0), py_n_launch_plane.at(1), py_n_launch_plane.at(2) };
                flight_plan.add_launch_plane_constraint(
                    py_p["n_launch_direction"].cast<double>(), n_launch_plane);
            }

            flight_plan.init_model();

            if (!py_p["conic_solution"].is_none())
            {
                py::array_t<double> py_xa = py_p["conic_solution"];
                std::vector<double> xa;
                for (int i = 0; i < py_xa.size(); i++)
                {
                    xa.push_back(py_xa.at(i));
                }
                flight_plan.set_conic_solution(xa);
            }

            try
            {
                if (py_p["run_model"].cast<bool>())
                {
                    flight_plan.run_model(py_p["num_evals"].cast<int>(), py_p["eps"].cast<double>(), py_p["eps_t"].cast<double>(), py_p["eps_x"].cast<double>());
                }
            }
            catch (const std::exception& e)
            {
                std::cerr << "Error during run model: " << e.what() << '\n';
            }

            ConicLunarFlightPlan::Result result = flight_plan.output_result(py_p["eps"].cast<double>());

            return py_output_result(result);
        }
        catch (const std::exception& e)
        {
            std::cerr << e.what() << '\n';
            return py::dict();
        }
    }

    Orbit craft_orbit;
    Orbit moon_orbit;
    double soi = 0.0;

    double myfunc(unsigned n, const double* x, double* grad, void* my_func_data)
    {
        Eigen::Vector3d rc = craft_orbit.get_position(x[0]);
        Eigen::Vector3d rm = moon_orbit.get_position(x[0]);


        return (rc - rm).norm() - soi;
    }

    py::array_t<double> relative_state(py::dict py_p)
    {
        py::array_t<double> output({ 100, 2 });
        double* output_ptr = py_arr_ptr(output);

        craft_orbit = Orbit(
            py_p["craft"]["orbit"]["gravitational_parameter"].cast<double>(),		// gravitational_parameter
            py_p["craft"]["orbit"]["semi_major_axis"].cast<double>(),         	    // semi_major_axis
            py_p["craft"]["orbit"]["eccentricity"].cast<double>(),				    // eccentricity
            py_p["craft"]["orbit"]["inclination"].cast<double>(),				    // inclination
            py_p["craft"]["orbit"]["longitude_of_ascending_node"].cast<double>(),	// longitude_of_ascending_node
            py_p["craft"]["orbit"]["argument_of_periapsis"].cast<double>(),		    // argument_of_periapsis
            py_p["craft"]["orbit"]["mean_anomaly_at_epoch"].cast<double>(),		    // mean_anomaly_at_epoch
            py_p["craft"]["orbit"]["epoch"].cast<double>()                          // epoch
        );

        moon_orbit = Orbit(
            py_p["moon"]["orbit"]["gravitational_parameter"].cast<double>(),		// gravitational_parameter
            py_p["moon"]["orbit"]["semi_major_axis"].cast<double>(),         	    // semi_major_axis
            py_p["moon"]["orbit"]["eccentricity"].cast<double>(),				    // eccentricity
            py_p["moon"]["orbit"]["inclination"].cast<double>(),				    // inclination
            py_p["moon"]["orbit"]["longitude_of_ascending_node"].cast<double>(),	// longitude_of_ascending_node
            py_p["moon"]["orbit"]["argument_of_periapsis"].cast<double>(),		    // argument_of_periapsis
            py_p["moon"]["orbit"]["mean_anomaly_at_epoch"].cast<double>(),		    // mean_anomaly_at_epoch
            py_p["moon"]["orbit"]["epoch"].cast<double>()                           // epoch
        );

        double t0 = py_p["initial_time"].cast<double>();
        double dt = astrodynamics::tau * craft_orbit.get_time_scale() / 100.0;
        

        for (int i = 0; i < 100; i++)
        {
            double t = t0 + dt * i;
            Eigen::Vector3d rc = craft_orbit.get_position(t);
            Eigen::Vector3d vc = craft_orbit.get_velocity(t);
            Eigen::Vector3d rm = moon_orbit.get_position(t);
            Eigen::Vector3d vm = moon_orbit.get_velocity(t);

            double dr = (rc - rm).norm();
            double dv = (vc - vm).norm();

            output_ptr[2 * i + 0] = t;
            output_ptr[2 * i + 1] = dr;
        }

        nlopt::opt opt(nlopt::LN_BOBYQA, 1);

        std::vector<double> lb(1);
        std::vector<double> ub(1);
        lb[0] = t0; 
        ub[0] = t0 + 100.0 * dt;
        opt.set_lower_bounds(lb);
        opt.set_upper_bounds(ub);
        opt.set_min_objective(myfunc, NULL);

        opt.set_xtol_rel(1e-4);
        std::vector<double> x(1);
        std::vector<double> x1(1);
        x[0] = t0 + 50.0 * dt;
        double minf;

        try {
            nlopt::result result = opt.optimize(x, minf);
            std::cout << "found minimum at f(" << x[0] << ") = "
                << std::setprecision(10) << minf << std::endl;
            std::cout << "num evals: " << opt.get_numevals() << std::endl;
        }
        catch (std::exception& e) {
            std::cout << "nlopt failed: " << e.what() << std::endl;
        }

        soi = py_p["moon"]["soi"].cast<double>();

        std::cout << soi - minf << '\n';

        if (minf < soi)
        { 
            ub[0] = x[0];
            x[0] = (t0 + x[0]) / 2.0;
            int num_evals = 0;
            double f;

            for (int i = 0; i < 20; i++)
            {
                num_evals++;
                f = myfunc(1, x.data(), NULL, NULL);
                x1[0] = x[0] + 1e-4;
                double f1 = myfunc(1, x1.data(), NULL, NULL);

                double xnew = x[0] - 1e-4 * f / (f1 - f);

                if (xnew > ub[0])
                {
                    x[0] = (x[0] + ub[0]) / 2.0;
                    std::cout << xnew << " ub\n";
                }
                else if (xnew < lb[0])
                {
                    x[0] = (x[0] + lb[0]) / 2.0;
                    std::cout << xnew << " lb\n";
                }
                else
                {
                    x[0] = xnew;
                }

                if (abs(f) < 1e-8)
                {
                    break;
                }

                std::cout << std::setprecision(17) << "x = " << x[0] << " f = " << f << " f1 = " << f1 << '\n';
            }

            std::cout << "found minimum at f(" << x[0] << ") = "
                << std::setprecision(10) << f << std::endl;
            std::cout << "num evals: " << num_evals << std::endl;
        }


        return output;
    }

    // create different correction functions for within and outside of soi...

    void constraint_numerical_gradient(unsigned m, unsigned n, const double* x, double* grad, void* f_data,
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

    void objective_numerical_gradient(unsigned n, const double* x, double* grad,
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

    struct TCMData
    {
        astrodynamics::ConicBody* moon;
        astrodynamics::ConicBody* planet;
        std::vector<double> t;
        Eigen::Vector3d r0;
        Eigen::Vector3d v0;
        Eigen::Vector3d rf;
        Eigen::Vector3d vf;
        double eps;
    };

    void constraints_tcm(unsigned m, double* result, unsigned n, const double* x, double* grad, void* f_data)
    {
        TCMData* data = reinterpret_cast<TCMData*>(f_data);

        Eigen::Vector3d rp = data->r0;
        Eigen::Vector3d vp = data->v0;
        Eigen::Vector3d rm = data->rf - data->moon->orbit->get_position(data->t[data->t.size() - 1]);
        Eigen::Vector3d vm = data->vf - data->moon->orbit->get_velocity(data->t[data->t.size() - 1]);

        double tsoi;

        // within moon soi
        for (int i = data->t.size() - 1; i > 0; i--)
        {
            vm(0) -= x[3 * (i - 1) + 0];
            vm(1) -= x[3 * (i - 1) + 1];
            vm(2) -= x[3 * (i - 1) + 2];

            double t0 = data->t[i - 1];
            double t1 = data->t[i];
            double dtsoi = astrodynamics::hyperbolic_orbit_time_at_distance(rm, vm, data->moon->soi, data->moon->mu) - astrodynamics::hyperbolic_orbit_time_at_distance(rm, vm, rm.norm(), data->moon->mu);

            if (t1 - t0 > dtsoi)
            {
                tsoi = t1 - dtsoi;
                std::tie(rm, vm) = astrodynamics::kepler_s(rm, vm, tsoi - t1, data->moon->mu, data->eps);
                break;
            }

            std::tie(rm, vm) = astrodynamics::kepler_s(rm, vm, t0 - t1, data->moon->mu, data->eps);
        }

        // within planet soi
        for (int i = 0; i < data->t.size(); i++)
        {
            double t0 = data->t[i];
            double t1 = data->t[i + 1];

            if (t1 > tsoi)
            {
                t1 = tsoi;
                std::tie(rp, vp) = astrodynamics::kepler_s(rp, vp, tsoi - t0, data->planet->mu, data->eps);
                break;
            }

            std::tie(rp, vp) = astrodynamics::kepler_s(rp, vp, t1 - t0, data->planet->mu, data->eps);
            vp(0) += x[3 * i + 0];
            vp(1) += x[3 * i + 1];
            vp(2) += x[3 * i + 2];
        }

        Eigen::Vector3d dr = rp - (rm + data->moon->orbit->get_position(tsoi));
        Eigen::Vector3d dv = vp - (vm + data->moon->orbit->get_velocity(tsoi));

        result[0] = dr(0) / data->moon->orbit->get_distance_scale();
        result[1] = dr(1) / data->moon->orbit->get_distance_scale();
        result[2] = dr(2) / data->moon->orbit->get_distance_scale();
        result[3] = dv(0) / data->moon->orbit->get_velocity_scale();
        result[4] = dv(1) / data->moon->orbit->get_velocity_scale();
        result[5] = dv(2) / data->moon->orbit->get_velocity_scale();

        if (grad)
        {
            constraint_numerical_gradient(m, n, x, grad, f_data, constraints_tcm);
        }
    }

    double objective_tcm(unsigned n, const double* x, double* grad, void* f_data)
    {
        TCMData* data = reinterpret_cast<TCMData*>(f_data);

        double f = 0.0;
        for (int i = 0; i < n / 3; i++)
        {
            f += sqrt(
                x[3 * i + 0] * x[3 * i + 0] +
                x[3 * i + 1] * x[3 * i + 1] +
                x[3 * i + 2] * x[3 * i + 2]
            );
        }

        if (grad)
        {
            objective_numerical_gradient(n, x, grad, f_data, objective_tcm);
        }

        return f;
    }

    py::tuple trajectory_correction(py::array_t<double> py_r, py::array_t<double> py_v, py::array_t<double> py_rt, py::array_t<double> py_vt, py::array_t<double> py_t, py::dict py_p)
    {
        try
        {
            Orbit planet_orbit = Orbit(
                py_p["planet"]["orbit"]["gravitational_parameter"].cast<double>(),		// gravitational_parameter
                py_p["planet"]["orbit"]["semi_major_axis"].cast<double>(),         	    // semi_major_axis
                py_p["planet"]["orbit"]["eccentricity"].cast<double>(),				    // eccentricity
                py_p["planet"]["orbit"]["inclination"].cast<double>(),				    // inclination
                py_p["planet"]["orbit"]["longitude_of_ascending_node"].cast<double>(),	// longitude_of_ascending_node
                py_p["planet"]["orbit"]["argument_of_periapsis"].cast<double>(),		    // argument_of_periapsis
                py_p["planet"]["orbit"]["mean_anomaly_at_epoch"].cast<double>(),		    // mean_anomaly_at_epoch
                py_p["planet"]["orbit"]["epoch"].cast<double>()                          // epoch
            );

            Orbit moon_orbit = Orbit(
                py_p["moon"]["orbit"]["gravitational_parameter"].cast<double>(),		// gravitational_parameter
                py_p["moon"]["orbit"]["semi_major_axis"].cast<double>(),         	    // semi_major_axis
                py_p["moon"]["orbit"]["eccentricity"].cast<double>(),				    // eccentricity
                py_p["moon"]["orbit"]["inclination"].cast<double>(),				    // inclination
                py_p["moon"]["orbit"]["longitude_of_ascending_node"].cast<double>(),	// longitude_of_ascending_node
                py_p["moon"]["orbit"]["argument_of_periapsis"].cast<double>(),		    // argument_of_periapsis
                py_p["moon"]["orbit"]["mean_anomaly_at_epoch"].cast<double>(),		    // mean_anomaly_at_epoch
                py_p["moon"]["orbit"]["epoch"].cast<double>()                           // epoch
            );

            TCMData tcm_data;

            astrodynamics::ConicBody moon = { &moon_orbit, py_p["moon"]["gravitational_parameter"].cast<double>(), py_p["moon"]["soi"].cast<double>(), py_p["moon"]["radius"].cast<double>() };
            astrodynamics::ConicBody planet = { &planet_orbit, py_p["planet"]["gravitational_parameter"].cast<double>(), py_p["planet"]["soi"].cast<double>(), py_p["planet"]["radius"].cast<double>() };

            tcm_data.moon = &moon;
            tcm_data.planet = &planet;

            for (int i = 0; i < py_t.size(); i++)
            {
                tcm_data.t.push_back(py_t.at(i));
            }

            tcm_data.r0(0) = py_r.at(0);
            tcm_data.r0(1) = py_r.at(1);
            tcm_data.r0(2) = py_r.at(2);

            tcm_data.v0(0) = py_v.at(0);
            tcm_data.v0(1) = py_v.at(1);
            tcm_data.v0(2) = py_v.at(2);

            tcm_data.rf(0) = py_rt.at(0);
            tcm_data.rf(1) = py_rt.at(1);
            tcm_data.rf(2) = py_rt.at(2);

            tcm_data.vf(0) = py_vt.at(0);
            tcm_data.vf(1) = py_vt.at(1);
            tcm_data.vf(2) = py_vt.at(2);

            tcm_data.eps = py_p["eps"].cast<double>();

            std::cout << moon.orbit->get_position(tcm_data.t[0]).transpose() << '\n';

            int m = 6;
            int n = (tcm_data.t.size() - 1) * 3;
            std::vector<double> x0(n, tcm_data.eps);

            std::vector<double> tol(m, tcm_data.eps);
            double minf;

            nlopt::opt opt = nlopt::opt("LD_SLSQP", n);
            opt.set_min_objective(objective_tcm, &tcm_data);
            opt.add_equality_mconstraint(constraints_tcm, &tcm_data, tol);
            opt.set_ftol_abs(tcm_data.eps);

            try
            {
                opt.optimize(x0, minf);
            }
            catch (const std::exception& e)
            {
                std::cerr << "Error during optimization: " << e.what() << '\n';
            }

            std::cout << std::setprecision(17);
            std::cout << "last optimum value: " << opt.last_optimum_value() << '\n';
            std::cout << "last optimum result: " << opt.last_optimize_result() << '\n';
            std::cout << "num evals: " << opt.get_numevals() << '\n';

            std::vector<double> result(6);

            constraints_tcm(m, result.data(), n, x0.data(), NULL, &tcm_data);

            py::list py_list_x0;

            for (int i = 0; i < x0.size() / 3; i++)
            {
                py::array_t<double> dv(3);
                auto mdv = dv.mutable_unchecked();
                mdv(0) = x0[3 * i + 0];
                mdv(1) = x0[3 * i + 1];
                mdv(2) = x0[3 * i + 2];
                py_list_x0.append(dv);
            }

            return py_list_x0;
        }
        catch (std::exception& e)
        {
            std::cerr << e.what() << "\n";
        }
    }
}