#include "py_lib.h"


PYBIND11_MODULE(ksplib, m) {
    m.doc() = "ksp library module.";
    // jdate functions
    py::module_ sm_jd = m.def_submodule("jd", "julian date module");
    sm_jd.def("to_jdate_from_ktime", &jdate::to_jdate_from_ktime, "kerbal time"_a);
    sm_jd.def("to_ktime_from_jdate", &jdate::to_ktime_from_jdate, "julian date"_a);
    sm_jd.def("to_jdate_from_date_time", &jdate::to_jdate_from_date_time, "month"_a, "day"_a, "year"_a, "time"_a);
    sm_jd.def("get_month_day_year", &jdate::get_month_day_year, "julian date"_a);
    sm_jd.def("get_time", &jdate::get_time, "julian date"_a);

    // ephemeris functions
    py::module sm_eph = m.def_submodule("eph", "ephemeris module");
    sm_eph.def("build_ephemeris", &ephemeris::build_ephemeris, "info"_a, "bodies"_a);
    sm_eph.def("load_ephemeris", &ephemeris::load_ephemeris, "file_name"_a);

    // astrodynamics functions
    py::module sm_astro = m.def_submodule("astro", "astrodynamics module");
    sm_astro.def("kepler", &astrodynamics::py_kepler, "r0"_a, "v0"_a, "t"_a, "mu"_a, "eps"_a = 1e-8);
    sm_astro.def("lambert", &astrodynamics::py_lambert, "r0"_a, "r1"_a, "t"_a, "mu"_a, "eps"_a = 1e-8, "d"_a = 1.0, "n"_a = py::none());

    // kerbal guidance system functions
    py::module_ sm_kgs = m.def_submodule("kgs", "kerbal guidance system module");
    sm_kgs.def("constraint_residuals", &kerbal_guidance_system::kgs_constraint_residuals, "t"_a, "yi"_a, "events"_a, "p"_a, "relerr"_a, "abserr"_a);
    sm_kgs.def("simulate_atm_phase", &kerbal_guidance_system::kgs_simulate_atm_phase, "t"_a, "yi"_a, "events"_a, "p"_a, "relerr"_a, "abserr"_a);
    sm_kgs.def("simulate_vac_phase", &kerbal_guidance_system::kgs_simulate_vac_phase, "t"_a, "yi"_a, "events"_a, "p"_a, "relerr"_a, "abserr"_a);
    sm_kgs.def("output_time_series", &kerbal_guidance_system::kgs_output_time_series, "t"_a, "yi"_a, "events"_a, "p_atm"_a, "p_vac"_a, "relerr"_a, "abserr"_a);

    //sm_kgs.def("solve_atm_ivp", &kerbal_guidance_system::solve_atm_ivp, "t"_a, "tout"_a, "y"_a, "p"_a);
    //sm_kgs.def("solve_vac_ivp", &kerbal_guidance_system::solve_vac_ivp, "t"_a, "tout"_a, "y"_a, "p"_a);
    //sm_kgs.def("output_atm_ivp", &kerbal_guidance_system::output_atm_ivp, "t"_a, "tout"_a, "steps"_a, "y"_a, "p"_a);
    //sm_kgs.def("output_vac_ivp", &kerbal_guidance_system::output_vac_ivp, "t"_a, "tout"_a, "steps"_a, "y"_a, "p"_a);
    //sm_kgs.def("constraint_residuals", &kerbal_guidance_system::constraint_residuals, "t"_a, "x"_a, "y"_a, "p"_a, "c"_a, "a_limit"_a, "relerr"_a, "abserr"_a);
    //sm_kgs.def("output_time_series", &kerbal_guidance_system::output_time_series, "p"_a);
    // sm_kgs.def("solve_orbit_burn", &kerbal_guidance_system::solve_orbit_burn, "t"_a, "tout"_a, "y"_a, "p"_a);

    // flight plan functions
    py::module_ sm_intp = m.def_submodule("intp", "interplanetary flight plan module");
    sm_intp.def("interplanetary", &interplanetary::interplanetary, "params"_a, "mode"_a, "show_errors"_a = false);
    sm_intp.def("trajectory_correction", &interplanetary::trajectory_correction, "r0"_a, "v0"_a, "rf"_a, "vf"_a, "t"_a, "bref"_a, "body_info"_a);

    // lunar flight plan functions
    py::module_ sm_lunar = m.def_submodule("lunar", "lunar flight plan module");
    sm_lunar.def("lunar", &lunar::lunar, "params"_a);

    // conic functions
    py::module_ sm_conic = m.def_submodule("conic", "conic module");
    sm_conic.def("lunar", &conic::lunar, "params"_a);
    sm_conic.def("relative_state", &conic::relative_state, "params"_a);
    sm_conic.def("trajectory_correction", &conic::trajectory_correction, "r0"_a, "v0"_a, "rf"_a, "vf"_a, "t"_a, "p"_a);
}
