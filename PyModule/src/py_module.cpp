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
    sm_astro.def("splev", &astrodynamics::splev, "x"_a, "tck"_a);
    sm_astro.def("bisplev", &astrodynamics::bisplev, "x"_a, "y"_a, "tck"_a);

    // kerbal guidance system functions
    py::module_ sm_kgs = m.def_submodule("kgs", "kerbal guidance system module");
    sm_kgs.def("solve_atm_ivp", &kerbal_guidance_system::solve_atm_ivp, "t"_a, "tout"_a, "y"_a, "p"_a);
    sm_kgs.def("solve_vac_ivp", &kerbal_guidance_system::solve_vac_ivp, "t"_a, "tout"_a, "y"_a, "p"_a);
    sm_kgs.def("output_atm_ivp", &kerbal_guidance_system::output_atm_ivp, "t"_a, "tout"_a, "steps"_a, "y"_a, "p"_a);
    sm_kgs.def("output_vac_ivp", &kerbal_guidance_system::output_vac_ivp, "t"_a, "tout"_a, "steps"_a, "y"_a, "p"_a);
    sm_kgs.def("solve_orbit_burn", &kerbal_guidance_system::solve_orbit_burn, "t"_a, "tout"_a, "y"_a, "p"_a);

    // flight plan functions
    py::module_ sm_intp = m.def_submodule("intp", "interplanetary flight plan module");
    sm_intp.def("interplanetary", &interplanetary::interplanetary, "params"_a, "mode"_a);
    sm_intp.def("trajectory_correction", &interplanetary::trajectory_correction, "r0"_a, "v0"_a, "rf"_a, "vf"_a, "t"_a, "bref"_a, "body_info"_a);
}
