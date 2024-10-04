#include "py_lib.h"


PYBIND11_MODULE(ksplib, m) {
    m.doc() = "ksp library module.";

    // kerbal guidance system functions
    py::module_ sm_kgs = m.def_submodule("kgs", "kerbal guidance system module");
    sm_kgs.def("simulate_atm_phase", &kerbal_guidance_system::simulate_atm_phase, "t"_a, "yi"_a, "events"_a, "p"_a);
    //sm_kgs.def("simulate_vac_phase", &kerbal_guidance_system::kgs_simulate_vac_phase, "t"_a, "yi"_a, "events"_a, "p"_a);
    //sm_kgs.def("constraint_residuals", &kerbal_guidance_system::kgs_constraint_residuals, "t"_a, "yi"_a, "events"_a, "p"_a);
    //sm_kgs.def("output_time_series", &kerbal_guidance_system::kgs_output_time_series, "t"_a, "yi"_a, "events"_a, "p_atm"_a, "p_vac"_a);
    //sm_kgs.def("test", &kerbal_guidance_system::test);
}
