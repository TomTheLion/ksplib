#pragma once

namespace astrodynamics
{
    //
    // variables
    // 

    constexpr double pi = 3.14159265358979323;
    constexpr double tau = 6.28318530717958647;

    //
    // structs
    //
    struct NBodyParams
    {
        int num_bodies;
        double muk;
        std::vector<double> mu;
    };

    struct NBodyEphemerisParams
    {
        Ephemeris* ephemeris;
    };

    struct NBodyRefEphemerisParams
    {
        Ephemeris* ephemeris;
        int reference_body;
        double reference_mu_scale;
        double reference_time_scale;
        double reference_distance_scale;
        double reference_velocity_scale;
    };

    //
    // math functions
    //

    Eigen::Vector3d lhc_to_rhc(Eigen::Vector3d a);
    double vector_angle(Eigen::Vector3d a, Eigen::Vector3d b, Eigen::Vector3d n = Eigen::Vector3d::Zero());
    double safe_acos(double x);
    std::tuple<double, double> stumpff(double z);

    //
    // orbital element functions
    //

    double orbit_semi_major_axis(Eigen::Vector3d r, Eigen::Vector3d v, double mu);
    double orbit_eccentricity(Eigen::Vector3d r, Eigen::Vector3d v, double mu);
    double orbit_inclination(Eigen::Vector3d r, Eigen::Vector3d v);
    double orbit_longitude_of_ascending_node(Eigen::Vector3d r, Eigen::Vector3d v);
    double orbit_argument_of_periapsis(Eigen::Vector3d r, Eigen::Vector3d v, double mu);
    double orbit_true_anomaly(Eigen::Vector3d r, Eigen::Vector3d v, double mu);

    //
    // additional orbital property functions
    //

    double orbit_apoapsis(Eigen::Vector3d r, Eigen::Vector3d v, double mu);
    double orbit_eccentric_anomaly(double f, double e);
    double orbit_mean_anomaly(double f, double e);

    //
    // flyby functions
    //

    // returns velocity after planetary flyby around axis normal to intial velocity
    // and planet's velocity
    Eigen::Vector3d planet_flyby(
        Eigen::Vector3d vb, Eigen::Vector3d v, double flyby_distance, double mu, bool increase);

    // returns state at periapsis of flyby trajectory around axis normal to initial
    // velocity and planet's velocity
    std::tuple<Eigen::Vector3d, Eigen::Vector3d> planet_orbit_periapsis(
        Ephemeris* ephemeris, Eigen::Vector3d v, double periapsis_distance, int direction, int b, double t);

    // returns state at periapsis of flyby trajectory when inbound and outbound
    // velocities are known
    std::tuple<Eigen::Vector3d, Eigen::Vector3d> planet_flyby_periapsis(
        Ephemeris* ephemeris, Eigen::Vector3d v0, Eigen::Vector3d v1, int b, double t);

    // returns time from periapsis to get to specified distance for a hyperbolic orbit
    double hyperbolic_orbit_time_at_distance(Eigen::Vector3d rp, Eigen::Vector3d vp, double d, double mu);

    //    
    // orbit propagation functions
    //

    // returns state after specified amount of time by solving universal Kepler's equation
    std::tuple<Eigen::Vector3d, Eigen::Vector3d> kepler(
        Eigen::Vector3d r0, Eigen::Vector3d v0, double t, double mu, double eps);

    // returns state after specified amount of time by solving universal Kepler's equation,
    // breaks trajectory into n time steps
    std::tuple<Eigen::Vector3d, Eigen::Vector3d> kepler_n(
        Eigen::Vector3d r0, Eigen::Vector3d v0, double t, double mu, double eps, int n);

    // returns state after specified amount of time by solving universal Kepler's equation,
    // breaks trajectory into n time steps where sqrt_mu_t is less than tau squared
    std::tuple<Eigen::Vector3d, Eigen::Vector3d> kepler_s(
        Eigen::Vector3d r0, Eigen::Vector3d v0, double t, double mu, double eps);

    // returns state after specified amount of time by solving universal Kepler's equation
    // additionally solves for the state transition matrix
    std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Matrix<double, 6, 1>, Eigen::Matrix<double, 6, 6>> kepler_stm(
        Eigen::Vector3d r0, Eigen::Vector3d v0, double t, double mu, double eps);

    // returns state after specified amount of time by solving universal Kepler's equation
    // breaks trajectory into n time steps where sqrt_mu_t is less than tau squared
    // additionally solves for the state transition matrix
    std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Matrix<double, 6, 1>, Eigen::Matrix<double, 6, 6>> kepler_stm_s(
        Eigen::Vector3d r0, Eigen::Vector3d v0, double t, double mu, double eps);

    // returns the initial and final velocity connecting two positions within a 
    // specified time interval in space by solving Lambert's problem
    std::tuple<Eigen::Vector3d, Eigen::Vector3d> lambert(
        Eigen::Vector3d r0, Eigen::Vector3d r1, double t, double mu, int d, Eigen::Vector3d n, double eps = 1e-8);

    //
    // reference frame functions
    //

    std::vector<double> state_add_body(Ephemeris* ephemeris, const double* x, int b, double t);
    std::vector<double> state_add_stm(const double* x);
    NBodyRefEphemerisParams reference_frame_params(Ephemeris* ephemeris, int b, double s);
    std::vector<double> state_to_reference_frame(NBodyRefEphemerisParams params, const double* x);

    //
    // numerical derivative functions
    //

    // n-body equation of motion relative to central body, mu from ephemeris
    void n_body_df(double t, double y[], double yp[], void* params);
    // n-body equation of motion relative to central body with vessel and state transition matrix
    void n_body_df_vessel(double t, double y[], double yp[], void* params);
    // n-body equation of motion relative to central body in ephemeris model
    void n_body_df_ephemeris(double t, double y[], double yp[], void* params);
    // n-body equation of motion relative to reference body in ephemeris model
    void n_body_df_ephemeris_ref(double t, double y[], double yp[], void* params);
    // n-body equation of motion relative to central body in ephemeris model
    // calculates state transition matrix and tau derivatives
    void n_body_df_stm_ephemeris(double t, double y[], double yp[], void* params);
    // n-body equation of motion relative to reference body in ephemeris model
    // calculates state transition matrix and tau derivatives
    void n_body_df_stm_ephemeris_ref(double t, double y[], double yp[], void* params);
}
