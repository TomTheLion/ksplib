cmake_minimum_required (VERSION 3.8)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")

project(ksplib)

add_subdirectory(pybind11)

include_directories(C:/dev/includes/eigen)
include_directories(C:/dev/includes/nlopt-2.7.1/src/api)
include_directories(nlopt)

include_directories(Spl/src)
include_directories(EquationALL/src)
include_directories(PyModule/src)
#[[
add_executable(
	testing
	EquationAll/src/rk/rk32.cpp
	EquationAll/src/rk/rk54.cpp
	EquationAll/src/rk/rk853.cpp
	EquationAll/src/ode/wrapode.cpp
	EquationAll/src/ode/ode.cpp
	EquationAll/src/rk/rk.cpp
	EquationAll/src/Equation.cpp
	EquationAll/src/testing.cpp
 )
 ]]
 
pybind11_add_module(
	ksplib
	Spl/src/Spl.cpp
	EquationAll/src/rk/rk32.cpp
	EquationAll/src/rk/rk54.cpp
	EquationAll/src/rk/rk853.cpp
	EquationAll/src/rk/rk.cpp
	EquationAll/src/ode/wrapode.cpp
	EquationAll/src/ode/ode.cpp
	EquationAll/src/Equation.cpp
	PyModule/src/py_lib.cpp
	PyModule/src/py_module.cpp
)

# target_link_libraries(ksplib PRIVATE C:/dev/includes/nlopt-2.7.1_build/Release/nlopt.lib)

# include_directories(C:/dev/includes/eigen)
# include_directories(C:/dev/includes/nlopt-2.7.1/src/api)

# include_directories(Equation/src)
# include_directories(EquationRK/src)
# include_directories(EquationALL/src)
# include_directories(Jdate/src)
# include_directories(Ephemeris/src)
# include_directories(Orbit/src)
# include_directories(astrodynamics/src)
# include_directories(Spl/src)

# include_directories(FlightPlan/src)
# include_directories(LunarFlightPlan/src)
# include_directories(ConicLunarFlightPlan/src)

# include_directories(PyModule/src)


# include_directories(testing/src)

#[[
add_executable(
	testing
	# Equation/src/ode.cpp
	# Equation/src/Equation.cpp
	EquationAll/src/rk32.cpp
	EquationAll/src/rk54.cpp
	EquationAll/src/rk853.cpp
	EquationAll/src/wrapode.cpp
	EquationAll/src/ode.cpp
	EquationAll/src/Equation.cpp
 "PyModule/src2/py_lib.cpp" "PyModule/src2/py_lib.h" "PyModule/src2/py_module.cpp"	"testing/testing.cpp"
)


add_executable(
	testing
	nlopt/nlopt.hpp
	Equation/src/Ode.cpp
	Equation/src/Equation.cpp
	Jdate/src/Jdate.cpp
	Ephemeris/src/Ephemeris.cpp
	Orbit/src/Orbit.cpp
	astrodynamics/src/astrodynamics.cpp
	FlightPlan/src/FlightPlan.cpp
	LunarFlightPlan/src/LunarFlightPlan.cpp
	ConicLunarFlightPlan/src/ConicLunarFlightPlan.cpp
	testing/src/testing.cpp
)

target_link_libraries(testing PRIVATE C:/dev/includes/nlopt-2.7.1_build/Release/nlopt.lib)


pybind11_add_module(
	ksplib
	nlopt/nlopt.hpp
	Jdate/src/Jdate.cpp
	Spl/src/Spl.cpp
	Equation/src/ode.cpp
	Equation/src/Equation.cpp
	Ephemeris/src/Ephemeris.cpp
	Orbit/src/Orbit.cpp
	astrodynamics/src/astrodynamics.cpp
	FlightPlan/src/FlightPlan.cpp
	LunarFlightPlan/src/LunarFlightPlan.cpp
	ConicLunarFlightPlan/src/ConicLunarFlightPlan.cpp
	PyModule/src/py_lib.cpp
	PyModule/src/py_module.cpp
)

target_link_libraries(ksplib PRIVATE C:/dev/includes/nlopt-2.7.1_build/Release/nlopt.lib)



add_subdirectory(pybind11)
include_directories(C:/dev/includes/eigen)

include_directories(Jdate/src)

include_directories(Ephemeris/src)
include_directories(Orbit/src)
include_directories(astrodynamics/src)
include_directories(FlightPlan/src)
include_directories(LunarFlightPlan/src)
include_directories(ConicLunarFlightPlan/src)
include_directories(Spl/src)
include_directories(nlopt)

include_directories(Equation/src)
include_directories(EquationRK/src)
include_directories(EquationALL/src)
include_directories(Jdate/src)
include_directories(Ephemeris/src)
include_directories(Orbit/src)
include_directories(astrodynamics/src)
include_directories(Spl/src)

include_directories(FlightPlan/src)
include_directories(LunarFlightPlan/src)
include_directories(ConicLunarFlightPlan/src)

include_directories(PyModule/src)



pybind11_add_module(
	ksplib
	nlopt/nlopt.hpp
	Jdate/src/Jdate.cpp
	Spl/src/Spl.cpp
	Equation/src/ode.cpp
	Equation/src/Equation.cpp
	Ephemeris/src/Ephemeris.cpp
	Orbit/src/Orbit.cpp
	astrodynamics/src/astrodynamics.cpp
	FlightPlan/src/FlightPlan.cpp
	LunarFlightPlan/src/LunarFlightPlan.cpp
	ConicLunarFlightPlan/src/ConicLunarFlightPlan.cpp
	PyModule/src/py_lib.cpp
	PyModule/src/py_module.cpp
)

target_link_libraries(ksplib PRIVATE C:/dev/includes/nlopt-2.7.1_build/Release/nlopt.lib)


add_executable (
	testing
	nlopt/nlopt.hpp
	Jdate/src/Jdate.cpp
	# Spl/src/Spl.cpp
	Equation/src/ode.cpp
	Equation/src/Equation.cpp
	Ephemeris/src/Ephemeris.cpp
	Orbit/src/Orbit.cpp
	astrodynamics/src/astrodynamics.cpp
	# FlightPlan/src/FlightPlan.cpp
	ConicLunarFlightPlan/src/ConicLunarFlightPlan.cpp
	ConicLunarFlightPlan/src/testing.cpp
)

target_link_libraries(testing PRIVATE C:/dev/includes/nlopt-2.7.1_build/Release/nlopt.lib)
]]