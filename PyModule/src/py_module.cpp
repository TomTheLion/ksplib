#include "py_lib.h"


PYBIND11_MODULE(ksplib, m) {
    m.doc() = "ksp library module.";
    // kerbal guidance system functions
    py::module_ sm_kgs = m.def_submodule("kgs", "kerbal guidance system module");
    sm_kgs.def("simulate_atm_phase", &kerbal_guidance_system::py_simulate_atm_phase, "t"_a, "yi"_a, "pitch_rate"_a, "azimuth"_a, "events"_a, "params"_a);
    sm_kgs.def("output_atm_phase", &kerbal_guidance_system::py_output_atm_phase, "t"_a, "yi"_a, "pitch_rate"_a, "azimuth"_a, "events"_a, "params"_a);
    sm_kgs.def("simulate_vac_phase", &kerbal_guidance_system::py_simulate_vac_phase, "t"_a, "yi"_a, "x"_a, "events"_a, "p"_a);
    sm_kgs.def("simulate_vac_phase_to_velocity", &kerbal_guidance_system::py_simulate_vac_phase_to_velocity, "t"_a, "yi"_a, "x"_a, "final_velocity"_a, "events"_a, "params"_a);
    sm_kgs.def("output_vac_phase", &kerbal_guidance_system::py_output_vac_phase, "t"_a, "yi"_a, "x"_a, "events"_a, "params"_a);
    sm_kgs.def("constraint_residuals", &kerbal_guidance_system::py_constraint_residuals, "t"_a, "yi"_a, "x"_a, "c"_a, "events"_a, "params"_a);
    sm_kgs.def("constraint_jacobian", &kerbal_guidance_system::py_constraint_jacobian, "t"_a, "yi"_a, "x"_a, "c"_a, "events"_a, "params"_a);
}