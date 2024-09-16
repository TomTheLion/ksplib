cmake_minimum_required (VERSION 3.8)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")

project(ksplib)

add_subdirectory(pybind11)

include_directories(C:/dev/includes/eigen)
include_directories(C:/dev/includes/nlopt-2.7.1/src/api)
include_directories(nlopt)

# include_directories(Equation/src)
# include_directories(EquationRK/src)
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


include_directories(testing/src)

add_executable(
	testing
	# Equation/src/ode.cpp
	# Equation/src/Equation.cpp
	EquationAll/src/dopr5.cpp
	EquationAll/src/Equation.cpp
	testing/src/testing2024.cpp
)

#[[
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
]]

#[[
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