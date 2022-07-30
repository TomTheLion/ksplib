#include <string>
#include <tuple>

#include <Eigen/Dense>

#include "Ephemeris.h"
#include "astrodynamics.h"

namespace astrodynamics
{
	//
	// math functions
	//

	Eigen::Vector3d lhc_to_rhc(Eigen::Vector3d a)
    {
        return { a(0), a(2), a(1) };
    }

	double vector_angle(Eigen::Vector3d a, Eigen::Vector3d b, Eigen::Vector3d n)
    {
        double angle = safe_acos(a.dot(b) / a.norm() / b.norm());

		if (n != Eigen::Vector3d::Zero() && a.cross(b).dot(n) < 0)
        {
			angle = tau - angle;
        }

		return angle;
    }

    double safe_acos(double x)
	{
		if (x > 1)
        {
			return 0.0;
        }
		else if (x < -1)
        {
			return pi;
        }
		else
        {
			return acos(x);
        }
	}

	std::tuple<double, double> stumpff(double z)
	{
		double s;
		double c;
		if (abs(z) < 1e-3)
		{
			s = (1.0 - z * (0.05 - z / 840.0)) / 6.0;
			c = 0.5 - z * (1.0 - z / 30.0) / 24.0;
		}
		else
		{
			if (z > 0)
			{
				double x = sqrt(z);
				s = (1.0 - sin(x) / x) / z;
				c = (1.0 - cos(x)) / z;
			}
			else
			{
				double x = sqrt(-z);
				s = (1.0 - sinh(x) / x) / z;
				c = (1.0 - cosh(x)) / z;
			}
		}
		return { s, c };
	}

	//
	// orbital element functions
	//

	double orbit_semi_major_axis(Eigen::Vector3d r, Eigen::Vector3d v, double mu)
    {
         return -mu / 2.0 / (v.squaredNorm() / 2.0 - mu / r.norm());
    }

    double orbit_eccentricity(Eigen::Vector3d r, Eigen::Vector3d v, double mu)
    {
        return (v.cross(r.cross(v)) / mu - r.normalized()).norm();
    }

	double orbit_inclination(Eigen::Vector3d r, Eigen::Vector3d v)
	{
		r = lhc_to_rhc(r);
		v = lhc_to_rhc(v);
		Eigen::Vector3d h = r.cross(v);
		return safe_acos(h(2) / h.norm());
	}

	double orbit_longitude_of_ascending_node(Eigen::Vector3d r, Eigen::Vector3d v)
	{
		r = lhc_to_rhc(r);
		v = lhc_to_rhc(v);
		Eigen::Vector3d h = r.cross(v);
		Eigen::Vector3d n = Eigen::Vector3d::UnitZ().cross(h);

		if (n(1) < 0)
		{
			return tau - safe_acos(n(0) / n.norm());
		}
		else
		{
			return safe_acos(n(0) / n.norm());
		}
	}

	double orbit_argument_of_periapsis(Eigen::Vector3d r, Eigen::Vector3d v, double mu)
	{
		r = lhc_to_rhc(r);
		v = lhc_to_rhc(v);
		Eigen::Vector3d h = r.cross(v);
		Eigen::Vector3d n = Eigen::Vector3d::UnitZ().cross(h);
		Eigen::Vector3d e = v.cross(r.cross(v)) / mu - r.normalized();

		if (e(2) < 0)
		{
			return tau - safe_acos(n.dot(e) / n.norm() / e.norm());
		}
		else
		{
			return safe_acos(n.dot(e) / n.norm() / e.norm());
		}
	}

	double orbit_true_anomaly(Eigen::Vector3d r, Eigen::Vector3d v, double mu)
	{
		r = lhc_to_rhc(r);
		v = lhc_to_rhc(v);
		Eigen::Vector3d e = v.cross(r.cross(v)) / mu - r.normalized();

		if (r.dot(v) < 0)
		{
			return tau - safe_acos(e.dot(r) / e.norm() / r.norm());
		}
		else
		{
			return safe_acos(e.dot(r) / e.norm() / r.norm());
		}	
	}

	//
    // additional orbital property functions
	//

    double orbit_apoapsis(Eigen::Vector3d r, Eigen::Vector3d v, double mu)
    {
        return orbit_semi_major_axis(r, v, mu) * (1.0 + orbit_eccentricity(r, v, mu));
    }

    double orbit_eccentric_anomaly(double f, double e)
    {
        if (e < 1)
        {
            return 2.0 * atan(sqrt((1.0 - e) / (1.0 + e)) * tan(f / 2.0));
        }
        else
        {
            return 2.0 * atanh(sqrt((e - 1.0) / (e + 1.0)) * tan(f / 2.0));
        }    
    }

    double orbit_mean_anomaly(double f, double e)
    {
        double eccentric_anomaly = orbit_eccentric_anomaly(f, e);

        if (e < 1)
        {
            return eccentric_anomaly - e * sin(eccentric_anomaly);
        }
        else
        {
            return e * sinh(eccentric_anomaly) - eccentric_anomaly;
        }    
    }

    //
    // flyby functions
    //

	// returns velocity after planetary flyby around axis normal to intial velocity
	// and planet's velocity
	// vb = velocity of body
	// v = initial velocity
	// flyby_distance = flyby distance at periapsis
	// mu = gravitational parameter of body
	// increase = if flyby is to increase velocity set to true, if flyby is to decrease
	// velocity set to false
    Eigen::Vector3d planet_flyby(
        Eigen::Vector3d vb, Eigen::Vector3d v, double flyby_distance, double mu, bool increase)
    {
        Eigen::Vector3d vi = v - vb;
        Eigen::Vector3d axis = vi.cross(vb).normalized();
        double semi_major_axis = -mu / vi.squaredNorm();
        double turn_angle = 2.0 * asin(1.0 / (1.0 - flyby_distance / semi_major_axis));

        Eigen::Vector3d v_plus = Eigen::AngleAxisd(turn_angle, axis) * vi + vb;
        Eigen::Vector3d v_minus = Eigen::AngleAxisd(-turn_angle, axis) * vi + vb;

		return v_plus.norm() > v_minus.norm() == increase ? v_plus : v_minus;
    }

	// returns state at periapsis of flyby trajectory around axis normal to initial
	// velocity and planet's velocity
	// v = initial velocity
	// direction = specifies if initial velocity is inbound or outbound veleocity
	// 1 = outbound, -1 = inbound
	// b = flyby body
	// t = flyby time
    std::tuple<Eigen::Vector3d, Eigen::Vector3d> planet_orbit_periapsis(
        Ephemeris* ephemeris, Eigen::Vector3d v, double periapsis_distance, int direction, int b, double t)
    {
        double mu = ephemeris->get_mu(b);
        auto [rb, vb] = ephemeris->get_position_velocity(b, t);

        Eigen::Vector3d vi = v - vb;

        Eigen::Vector3d n = rb.cross(vb).normalized();
        Eigen::Vector3d h_unit = vi.cross(vb).normalized();

        if (h_unit.dot(n) < 0) h_unit *= -1.0;
        
        double delta = 2.0 * asin(1.0 / (1.0 + periapsis_distance * vi.squaredNorm() / mu));

        Eigen::Vector3d r_unit = direction * (Eigen::AngleAxisd(-direction * 0.5 * (pi + delta), h_unit) * vi.normalized());
        Eigen::Vector3d v_unit = h_unit.cross(r_unit);

        Eigen::Vector3d rp = periapsis_distance * r_unit;
        Eigen::Vector3d vp = sqrt(2.0 * mu / periapsis_distance + vi.squaredNorm()) * v_unit;

        return { rp, vp };
    }

	// returns state at periapsis of flyby trajectory when inbound and outbound
	// velocities are known
	// v0 = inbound velocity
	// v1 = outbound velocity
	// b = flyby body
	// t = flyby time
    std::tuple<Eigen::Vector3d, Eigen::Vector3d> planet_flyby_periapsis(
		Ephemeris* ephemeris, Eigen::Vector3d v0, Eigen::Vector3d v1, int b, double t)
    {
        double mu = ephemeris->get_mu(b);
        auto [rb, vb] = ephemeris->get_position_velocity(b, t);
        Eigen::Vector3d vi0 = v0 - vb;
        Eigen::Vector3d vi1 = v1 - vb;

        Eigen::Vector3d r_unit = (vi0 - vi1).normalized();
        Eigen::Vector3d h_unit = vi0.cross(vi1).normalized();
        Eigen::Vector3d v_unit = h_unit.cross(r_unit);

        double delta = acos(vi0.dot(vi1) / vi0.norm() / vi1.norm());

        Eigen::Vector3d rp = mu / vi1.squaredNorm() * (1.0 / sin(delta / 2.0) - 1.0) * r_unit;
        Eigen::Vector3d vp = sqrt(2.0 * mu / rp.norm() + vi1.squaredNorm()) * v_unit;

        return { rp, vp };
    }

	// returns time from periapsis to get to specified distance for a hyperbolic orbit
	// rp = periapsis position
	// vp = periapsis velocity
	// d = distance from body
	// mu = gravitational parameter of body
    double hyperbolic_orbit_time_at_distance(Eigen::Vector3d rp, Eigen::Vector3d vp, double d, double mu)
    {
        double e = orbit_eccentricity(rp, vp, mu);
        double a = orbit_semi_major_axis(rp, vp, mu);
        double h = acosh((-d / a + 1.0) / e);

        return (e * sinh(h) - h) / sqrt(mu / -pow(a, 3));
    }

	//
    // orbit propagation functions
	//

	// solve universal Kepler's equation using Newton's method
    void kepler_solve(
        double r0, double alpha, double sigma, double t, double mu, double eps,
		double& x, double& x2, double& z, double& s, double& c, double& x2c
    )
    {
		double beta = 1.0 - alpha * r0;
		double sqrt_mu_t = sqrt(mu) * t;

		x = sqrt_mu_t * abs(alpha);
		double f, df, df2, dx;

		for (int iter = 0; iter < 100; iter++)
		{
			x2 = x * x;
			z = alpha * x2;
			std::tie(s, c) = stumpff(z);
			x2c = x2 * c;
			f = sigma * x2c + beta * x * x2 * s + r0 * x - sqrt_mu_t;
			df = sigma * x * (1.0 - z * s) + beta * x2c + r0;
			df2 = sigma * (1.0 - z * c) + beta * x * (1.0 - z * s);

			dx = -5.0 * f / (df * (1.0 + sqrt(abs(16.0 - 20.0 * f * df2 / (df * df)))));
			x += dx;

			if (abs(f) < eps || abs(dx) < eps)
            {
                return;
            }

            if (isnan(x))
            {
                throw std::runtime_error("\nRuntime Error: \"kepler\" failed to converge, x is nan");
            }
		}

        throw std::runtime_error("\nRuntime Error: \"kepler\" failed to converge in 100 iterations");
    }

	// returns state after specified amount of time by solving universal Kepler's equation
	// r0 = initial position
	// v0 = initial velocity
	// t = delta time
	// mu = gravitational parameter of body
	// eps = convergence tolerance
    std::tuple<Eigen::Vector3d, Eigen::Vector3d> kepler(
        Eigen::Vector3d r0, Eigen::Vector3d v0, double t, double mu, double eps)
    {
        double r0n = r0.norm();
		double alpha = 2.0 / r0n - v0.squaredNorm() / mu;
		double sigma = r0.dot(v0) / sqrt(mu);
		double x, x2, z, s, c, x2c;

        kepler_solve(r0n, alpha, sigma, t, mu, eps, x, x2, z, s, c, x2c);

        Eigen::Vector3d r1 = (1.0 - x2c / r0n) * r0 + (t - x * x2 * s / sqrt(mu)) * v0;
		Eigen::Vector3d v1 = (x * sqrt(mu) / r0n / r1.norm() * (z * s - 1.0)) * r0 + (1.0 - x2c / r1.norm()) * v0;

        return { r1, v1 };
    }

	// returns state after specified amount of time by solving universal Kepler's equation,
	// breaks trajectory into n time steps
	// r0 = initial position
	// v0 = initial velocity
	// t = delta time
	// mu = gravitational parameter of body
	// eps = convergence tolerance
	// n = number of time steps
    std::tuple<Eigen::Vector3d, Eigen::Vector3d> kepler_n(
        Eigen::Vector3d r0, Eigen::Vector3d v0, double t, double mu, double eps, int n)
    {
        t /= n;

        for (int i = 0; i < n; i++)
        {
            std::tie(r0, v0) = kepler(r0, v0, t, mu, eps);
        }

        return { r0, v0 };
    }

	// returns state after specified amount of time by solving universal Kepler's equation,
	// breaks trajectory into n time steps where sqrt_mu_t is less than tau squared
	// r0 = initial position
	// v0 = initial velocity
	// t = delta time
	// mu = gravitational parameter of body
	// eps = convergence tolerance
	std::tuple<Eigen::Vector3d, Eigen::Vector3d> kepler_s(
		Eigen::Vector3d r0, Eigen::Vector3d v0, double t, double mu, double eps)
	{
		double alpha = 2.0 / r0.norm() - v0.squaredNorm() / mu;
		int n = sqrt(abs(pow(alpha, 3) * pow(t, 2) * mu)) / tau + 1;
		t /= n;

		for (int i = 0; i < n; i++)
		{
			std::tie(r0, v0) = kepler(r0, v0, t, mu, eps);
		}

		return { r0, v0 };
	}

	// returns state after specified amount of time by solving universal Kepler's equation
	// additionally solves for the state transition matrix
	// r0 = initial position
	// v0 = initial velocity
	// t = delta time
	// mu = gravitational parameter of body
	// eps = convergence tolerance
    std::tuple<Eigen::Vector3d, Eigen::Vector3d, Eigen::Matrix<double, 6, 1>, Eigen::Matrix<double, 6, 6>> kepler_stm(
        Eigen::Vector3d r0, Eigen::Vector3d v0, double t, double mu, double eps)
    {
        double r0n = r0.norm();
		double alpha = 2.0 / r0n - v0.squaredNorm() / mu;
		double sigma = r0.dot(v0) / sqrt(mu);
        double sqrt_mu_t = sqrt(mu) * t;
		double x, x2, z, s, c, x2c;

        kepler_solve(r0n, alpha, sigma, t, mu, eps, x, x2, z, s, c, x2c);

        double f = 1.0 - x2c / r0n;
        double g = t - x * x2 * s / sqrt(mu);
        Eigen::Vector3d r1 = f * r0 + g * v0;
        double r1n = r1.norm();
        double fdot = x * sqrt(mu) / r0n / r1n * (z * s - 1.0);
        double gdot = 1.0 - x2c / r1n;
        Eigen::Vector3d v1 = fdot * r0 + gdot * v0;

        double u2 = r0n * (1.0 - f);
        double u3 = sqrt(mu) * (t - g);
        double u4 = (x * x / 2.0 - u2) / alpha;
        double u5 = (x * x * x / 6.0 - u3) / alpha;
        double uc = (3.0 * u5 - x * u4 - sqrt_mu_t * u2) / sqrt(mu);

		double r0n3 = r0n * r0n * r0n;
		double r1n3 = r1n * r1n * r1n;
		Eigen::Vector3d dv = (v1 - v0);
		
		Eigen::Matrix3d drdr0 =
        r1n / mu * dv * dv.transpose() +
        (r0n * (1.0 - f) * r1 * r0.transpose() + uc * v1 * r0.transpose()) / r0n3 +
        f * Eigen::Matrix3d::Identity();
        
        Eigen::Matrix3d dvdv0 =
        r0n / mu * dv * dv.transpose() +
        (r0n * (1.0 - f) * r1 * r0.transpose() - uc * r1 * v0.transpose()) / r1n3 +
        gdot * Eigen::Matrix3d::Identity();

        Eigen::Matrix3d drdv0 =
        r0n / mu * (1.0 - f) * ((r1 - r0) * v0.transpose() - dv * r0.transpose()) + 
        uc / mu * v1 * v0.transpose() +
        g * Eigen::Matrix3d::Identity();

        Eigen::Matrix3d dvdr0 = 
        -dv * r0.transpose() / r0n / r0n +
        -r1 * dv.transpose() / r1n / r1n +
        fdot * (Eigen::Matrix3d::Identity() - r1 * r1.transpose() / r1n / r1n + (r1 * v1.transpose() - v1 * r1.transpose()) * r1 * dv.transpose() / mu / r1n) +
        - mu * uc * r1 * r0.transpose() / r1n3 / r0n3;

        Eigen::Matrix<double, 6, 6> stm;
        stm.block<3, 3>(0, 0) = drdr0;
        stm.block<3, 3>(0, 3) = drdv0;
        stm.block<3, 3>(3, 0) = dvdr0;
        stm.block<3, 3>(3, 3) = dvdv0;

		Eigen::Matrix<double, 6, 1> xp1;
		xp1 << v1(0), v1(1), v1(2), -mu * r1(0) / r1n3, -mu * r1(1) / r1n3, -mu * r1(2) / r1n3;

        return { r1, v1, xp1, stm };
    }

	// returns the initial and final velocity connecting two positions within a 
	// specified time interval in space by solving Lambert's problem
	// r0 = initial position
	// r1 = final position
	// t = delta time
	// mu = gravitational parameter of body
	// d = specified direction, 1 = prograde, -1 = retrograde
	// n = normal vector to reference plane defining direction
	// eps = convergence tolerance
    std::tuple<Eigen::Vector3d, Eigen::Vector3d> lambert(
        Eigen::Vector3d r0, Eigen::Vector3d r1, double t, double mu, int d, Eigen::Vector3d n, double eps)
	{
		double beta = 1.0;
		double r0n = r0.norm();
		double r1n = r1.norm();
		double mu_t = sqrt(mu) * t;
		double theta = safe_acos(r0.dot(r1) / r0n / r1n);

		if (d * r0.cross(r1).dot(n) < 0)
		{
			theta = tau - theta;
		}

		double a = sin(theta) * sqrt(r0n * r1n / (1.0 - cos(theta)));
		double z = 0.0;

		double f;
		double fnew;
		double df;

		auto fdf = [r0n, r1n, a, mu_t](double z, double &f, double &df)
		{
			auto [s, c] = stumpff(z);
			double y = r0n + r1n + a * (z * s - 1.0) / sqrt(c);
			f = pow(y / c, 1.5) * s + a * sqrt(y) - mu_t;

			if (abs(z) < 1e-3)
			{
				df = 0.035355339059327376 * pow(y, 1.5) + a * 0.125 * (sqrt(y) + a * sqrt(0.5 / y));
			}
			else
			{
				df = pow(y / c, 1.5) * (0.5 / z * (c - 1.5 * s / c) + 0.75 * s * s / c) + a * 0.125 * (3.0 * s / c * sqrt(y) + a * sqrt(c / y));
			}
		};

		fdf(z, f, df);

		for (int iter = 0; iter < 100; iter++)
		{
			double dz = -f / df;
			double max_dz = 19.739208802178716 - 0.5 * z;

			if (dz > max_dz)
            {
				dz = max_dz;
            }

			double znew = z + beta * dz;
			fdf(znew, fnew, df);

			if (abs(fnew) > abs(f))
			{
				double r = abs(fnew / f);
				beta *= (sqrt(1 + 6 * r) - 1) / (3 * r);
			}
			else
			{
				beta = 1.0;
				f = fnew;
				z = znew;
			}

			if (abs(f) < eps || abs(dz) < eps)
			{
				auto [s, c] = stumpff(z);
				double y = r0n + r1n + a * (z * s - 1.0) / sqrt(c);
				Eigen::Vector3d v0 = 1.0 / (a * sqrt(y / mu)) * (r1 - (1.0 - y / r0n) * r0);
				Eigen::Vector3d v1 = 1.0 / (a * sqrt(y / mu)) * ((1.0 - y / r1n) * r1 - r0);
				return { v0, v1 };
			}
		}

        throw std::runtime_error("\nRuntime Error: \"lambert's problem\" failed to converge in 100 iterations");
	}

    //
    // reference frame functions
    //

	// adds state vector to the state vector of a specified body
    std::vector<double> state_add_body(Ephemeris* ephemeris, const double* x, int b, double t)
    {
        auto [rb, vb] = ephemeris->get_position_velocity(b, t);

		std::vector<double> yi
		{
			x[0] + rb(0), 
            x[1] + rb(1), 
            x[2] + rb(2), 
            x[3] + vb(0), 
            x[4] + vb(1), 
            x[5] + vb(2)
		};

		return yi;
    }

	// appends initial conditions for state transition matrix and derivative of
	// final state with respect to epoch time to a state vector
	std::vector<double> state_add_stm(const double* x)
    {
        std::vector<double> yi
		{
			x[0], x[1], x[2], x[3], x[4], x[5], 
			1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0
		};

		return yi;
    }

	// changes the reference variables to reflect a different body and scale factor
	NBodyRefEphemerisParams reference_frame_params(Ephemeris* ephemeris, int b, double s)
	{
		double mu = ephemeris->get_mu(b);
		double d_scale = s;
		double t_scale = sqrt(pow(d_scale, 3) / mu);
		double v_scale = d_scale / t_scale;

		return NBodyRefEphemerisParams { ephemeris, b, mu, t_scale, d_scale, v_scale };
	}

	// converts the scale factor of a state vector from the sun centered frame to a
	// planet centered frame
    std::vector<double> state_to_reference_frame(NBodyRefEphemerisParams params, const double* x)
	{
		std::vector<double> xs(6);
		xs[0] = x[0] / params.reference_distance_scale;
		xs[1] = x[1] / params.reference_distance_scale;
		xs[2] = x[2] / params.reference_distance_scale;
		xs[3] = x[3] / params.reference_velocity_scale;
		xs[4] = x[4] / params.reference_velocity_scale;
		xs[5] = x[5] / params.reference_velocity_scale;

		return xs;
	}

    //
    // numerical derivative functions
    //

	// n-body equation of motion relative to central body, mu from ephemeris
    void n_body_df(double t, double y[], double yp[], void* params)
    {
		NBodyParams* p = reinterpret_cast<NBodyParams*>(params);

        double ypkx = 0;
        double ypky = 0;
        double ypkz = 0;
        
        for (int i = 0; i < p->num_bodies; i++)
        {
            const double* yi = y + 6 * i;
            double* ypi = yp + 6 * i;

			// r = rki
            const double r2 = yi[0] * yi[0] + yi[1] * yi[1] + yi[2] * yi[2];
            const double r3 = sqrt(r2) * r2;
            const double mur3i = p->mu[i] / r3;
            const double mur3k = p->muk / r3;

            ypkx += mur3i * yi[0];
            ypky += mur3i * yi[1];
            ypkz += mur3i * yi[2];

            ypi[0] = yi[3];
            ypi[1] = yi[4];
            ypi[2] = yi[5];
            ypi[3] = -mur3k * yi[0];
            ypi[4] = -mur3k * yi[1];
            ypi[5] = -mur3k * yi[2];
        }

        for (int i = 0; i < p->num_bodies; i++)
        {
            double* ypi = yp + 6 * i;

            ypi[3] -= ypkx;
            ypi[4] -= ypky;
            ypi[5] -= ypkz;
        }

        for (int i = 0; i < p->num_bodies - 1; i++)
        {
            const double* yi = y + 6 * i;
            double* ypi = yp + 6 * i;
            for (int j = i + 1; j < p->num_bodies; j++)
            {
                const double* yj = y + 6 * j;
                double* ypj = yp + 6 * j;

				// r = rij
                const double rx = yj[0] - yi[0];
                const double ry = yj[1] - yi[1];
                const double rz = yj[2] - yi[2];

                const double r2 = rx * rx + ry * ry + rz * rz;
                const double r3 = sqrt(r2) * r2;
                const double mur3i = p->mu[i] / r3;
                const double mur3j = p->mu[j] / r3;

                ypi[3] += mur3j * rx;
                ypi[4] += mur3j * ry;
                ypi[5] += mur3j * rz;

                ypj[3] -= mur3i * rx;
                ypj[4] -= mur3i * ry;
                ypj[5] -= mur3i * rz;
            }
        }
    }

	// n-body equation of motion relative to central body with vessel and state transition matrix
	void n_body_df_vessel(double t, double y[], double yp[], void* params)
	{
		n_body_df(t, y + 42, yp + 42, params);
	
		NBodyParams* p = reinterpret_cast<NBodyParams*>(params);

		double rikx = y[0];
		double riky = y[1];
		double rikz = y[2];

		double rik2 = rikx * rikx + riky * riky + rikz * rikz;
		double rik3 = sqrt(rik2) * rik2;
		double murik3 = p->muk / rik3;
		double murik5 = murik3 / rik2;

		yp[0] = y[3];
		yp[1] = y[4];
		yp[2] = y[5];
		yp[3] = -murik3 * rikx;
		yp[4] = -murik3 * riky;
		yp[5] = -murik3 * rikz;

		double a41 = 3 * murik5 * rikx * rikx - murik3;
		double a52 = 3 * murik5 * riky * riky - murik3;
		double a63 = 3 * murik5 * rikz * rikz - murik3;
		double a42 = 3 * murik5 * rikx * riky;
		double a43 = 3 * murik5 * rikx * rikz;
		double a53 = 3 * murik5 * riky * rikz;

		for (int j = 0; j < p->num_bodies; j++)
		{
			const double* yj = y + 42 + 6 * j;

			double rjkx = yj[0];
			double rjky = yj[1];
			double rjkz = yj[2];

			double vjkx = yj[3];
			double vjky = yj[4];
			double vjkz = yj[5];

			double rjk2 = rjkx * rjkx + rjky * rjky + rjkz * rjkz;
			double rjk3 = sqrt(rjk2) * rjk2;
			double murjk3 = p->mu[j] / rjk3;
			double murjk5 = murjk3 / rjk2;

			double rijx = rjkx - y[0];
			double rijy = rjky - y[1];
			double rijz = rjkz - y[2];

			double rij2 = rijx * rijx + rijy * rijy + rijz * rijz;
			double rij3 = sqrt(rij2) * rij2;
			double murij3 = p->mu[j] / rij3;
			double murij5 = murij3 / rij2;

			yp[3] += murij3 * rijx - murjk3 * rjkx;
			yp[4] += murij3 * rijy - murjk3 * rjky;
			yp[5] += murij3 * rijz - murjk3 * rjkz;

			double xxij = 3 * murij5 * rijx * rijx - murij3;
			double yyij = 3 * murij5 * rijy * rijy - murij3;
			double zzij = 3 * murij5 * rijz * rijz - murij3;
			double xyij = 3 * murij5 * rijx * rijy;
			double xzij = 3 * murij5 * rijx * rijz;
			double yzij = 3 * murij5 * rijy * rijz;

			double xxjk = 3 * murjk5 * rjkx * rjkx - murjk3;
			double yyjk = 3 * murjk5 * rjky * rjky - murjk3;
			double zzjk = 3 * murjk5 * rjkz * rjkz - murjk3;
			double xyjk = 3 * murjk5 * rjkx * rjky;
			double xzjk = 3 * murjk5 * rjkx * rjkz;
			double yzjk = 3 * murjk5 * rjky * rjkz;

			a41 += xxij;
			a52 += yyij;
			a63 += zzij;
			a42 += xyij;
			a43 += xzij;
			a53 += yzij;
		}

		const double* stm = y + 6;
		double* dstm = yp + 6;

		for (int i = 0; i < 6; i++)
		{
			dstm[6 * i + 0] = stm[6 * i + 3];
			dstm[6 * i + 1] = stm[6 * i + 4];
			dstm[6 * i + 2] = stm[6 * i + 5];
			dstm[6 * i + 3] = a41 * stm[6 * i] + a42 * stm[6 * i + 1] + a43 * stm[6 * i + 2];
			dstm[6 * i + 4] = a42 * stm[6 * i] + a52 * stm[6 * i + 1] + a53 * stm[6 * i + 2];
			dstm[6 * i + 5] = a43 * stm[6 * i] + a53 * stm[6 * i + 1] + a63 * stm[6 * i + 2];
		}
	}

	// n-body equation of motion relative to central body in ephemeris model
    void n_body_df_ephemeris(double t, double y[], double yp[], void* params)
	{
		NBodyEphemerisParams* p = reinterpret_cast<NBodyEphemerisParams*>(params);

        double rki2 = y[0] * y[0] + y[1] * y[1] + y[2] * y[2];
        double rki3 = sqrt(rki2) * rki2;
        double muk_rki3 = p->ephemeris->get_muk() / rki3;

		yp[0] = y[3];
		yp[1] = y[4];
		yp[2] = y[5];
		yp[3] = -muk_rki3 * y[0];
		yp[4] = -muk_rki3 * y[1];
		yp[5] = -muk_rki3 * y[2];

		for (int j = 0; j < p->ephemeris->get_num_bodies(); j++)
		{
			Eigen::Vector3d rkj = p->ephemeris->get_position(j, t);

			double rkjx = rkj(0);
			double rkjy = rkj(1);
			double rkjz = rkj(2);

			double rkj2 = rkjx * rkjx + rkjy * rkjy + rkjz * rkjz;
			double rkj3 = sqrt(rkj2) * rkj2;
			double muj_rkj3 = p->ephemeris->get_mu(j) / rkj3;

			double rijx = rkjx - y[0];
			double rijy = rkjy - y[1];
			double rijz = rkjz - y[2];

			double rij2 = rijx * rijx + rijy * rijy + rijz * rijz;
			double rij3 = sqrt(rij2) * rij2;
			double muj_rij3 = p->ephemeris->get_mu(j) / rij3;

			yp[3] += muj_rij3 * rijx - muj_rkj3 * rkjx;
			yp[4] += muj_rij3 * rijy - muj_rkj3 * rkjy;
			yp[5] += muj_rij3 * rijz - muj_rkj3 * rkjz;
		}
	}
	
	// n-body equation of motion relative to reference body in ephemeris model
	void n_body_df_ephemeris_ref(double t, double y[], double yp[], void* params)
	{
		NBodyRefEphemerisParams* p = reinterpret_cast<NBodyRefEphemerisParams*>(params);

		Eigen::Vector3d rk = p->ephemeris->get_position(p->reference_body, p->reference_time_scale * t);

        double rkix = y[0];
        double rkiy = y[1];
        double rkiz = y[2];

        double rki2 = rkix * rkix + rkiy * rkiy + rkiz * rkiz;
        double rki3 = sqrt(rki2) * rki2;
        double muk_rki3 = 1.0 / rki3;

		yp[0] = y[3];
		yp[1] = y[4];
		yp[2] = y[5];
		yp[3] = -muk_rki3 * rkix;
		yp[4] = -muk_rki3 * rkiy;
		yp[5] = -muk_rki3 * rkiz;

		for (int j = 0; j < p->ephemeris->get_num_bodies(); j++)
		{
			double muj;
			Eigen::Vector3d rkj;
			
			if (j == p->reference_body)
			{
				rkj = -rk / p->reference_distance_scale;
				muj = p->ephemeris->get_muk() / p->reference_mu_scale;
			}
			else
			{
				rkj = (p->ephemeris->get_position(j, p->reference_time_scale * t) - rk) / p->reference_distance_scale;
				muj = p->ephemeris->get_mu(j) / p->reference_mu_scale;
			}

            double rkjx = rkj(0);
			double rkjy = rkj(1);
			double rkjz = rkj(2);

            double rkj2 = rkjx * rkjx + rkjy * rkjy + rkjz * rkjz;
			double rkj3 = sqrt(rkj2) * rkj2;
			double muj_rkj3 = muj / rkj3;

			double rijx = rkjx - y[0];
			double rijy = rkjy - y[1];
			double rijz = rkjz - y[2];

			double rij2 = rijx * rijx + rijy * rijy + rijz * rijz;
			double rij3 = sqrt(rij2) * rij2;
			double muj_rij3 = muj / rij3;

			yp[3] += muj_rij3 * rijx - muj_rkj3 * rkjx;
			yp[4] += muj_rij3 * rijy - muj_rkj3 * rkjy;
			yp[5] += muj_rij3 * rijz - muj_rkj3 * rkjz;
		}
	}
	
	// n-body equation of motion relative to central body in ephemeris model
    // calculates state transition matrix and tau derivatives
	void n_body_df_stm_ephemeris(double t, double y[], double yp[], void* params)
	{
		NBodyEphemerisParams* p = reinterpret_cast<NBodyEphemerisParams*>(params);

		double rkix = y[0];
        double rkiy = y[1];
        double rkiz = y[2];

        double rki2 = rkix * rkix + rkiy * rkiy + rkiz * rkiz;
        double rki3 = sqrt(rki2) * rki2;
        double muk_rki3 = p->ephemeris->get_muk() / rki3;
        double muk_rki5 = muk_rki3 / rki2;

		yp[0] = y[3];
		yp[1] = y[4];
		yp[2] = y[5];
		yp[3] = -muk_rki3 * rkix;
		yp[4] = -muk_rki3 * rkiy;
		yp[5] = -muk_rki3 * rkiz;

		double a41 = 3 * muk_rki5 * rkix * rkix - muk_rki3;
		double a52 = 3 * muk_rki5 * rkiy * rkiy - muk_rki3;
		double a63 = 3 * muk_rki5 * rkiz * rkiz - muk_rki3;
		double a42 = 3 * muk_rki5 * rkix * rkiy;
		double a43 = 3 * muk_rki5 * rkix * rkiz;
		double a53 = 3 * muk_rki5 * rkiy * rkiz;

        double sx = 0;
        double sy = 0;
        double sz = 0;

		for (int j = 0; j < p->ephemeris->get_num_bodies(); j++)
		{
			auto [rkj, vkj] = p->ephemeris->get_position_velocity(j, t);

            double rkjx = rkj(0);
			double rkjy = rkj(1);
			double rkjz = rkj(2);

            double vkjx = vkj(0);
			double vkjy = vkj(1);
			double vkjz = vkj(2);

            double rkj2 = rkjx * rkjx + rkjy * rkjy + rkjz * rkjz;
			double rkj3 = sqrt(rkj2) * rkj2;
			double muj_rkj3 = p->ephemeris->get_mu(j) / rkj3;
			double muj_rkj5 = muj_rkj3 / rkj2;

			double rijx = rkjx - y[0];
			double rijy = rkjy - y[1];
			double rijz = rkjz - y[2];

			double rij2 = rijx * rijx + rijy * rijy + rijz * rijz;
			double rij3 = sqrt(rij2) * rij2;
			double muj_rij3 = p->ephemeris->get_mu(j) / rij3;
			double muj_rij5 = muj_rij3 / rij2;

			yp[3] += muj_rij3 * rijx - muj_rkj3 * rkjx;
			yp[4] += muj_rij3 * rijy - muj_rkj3 * rkjy;
			yp[5] += muj_rij3 * rijz - muj_rkj3 * rkjz;

            double xxij = 3 * muj_rij5 * rijx * rijx - muj_rij3;
            double yyij = 3 * muj_rij5 * rijy * rijy - muj_rij3;
            double zzij = 3 * muj_rij5 * rijz * rijz - muj_rij3;
            double xyij = 3 * muj_rij5 * rijx * rijy;
            double xzij = 3 * muj_rij5 * rijx * rijz;
            double yzij = 3 * muj_rij5 * rijy * rijz;

            double xxkj = 3 * muj_rkj5 * rkjx * rkjx - muj_rkj3;
            double yykj = 3 * muj_rkj5 * rkjy * rkjy - muj_rkj3;
            double zzkj = 3 * muj_rkj5 * rkjz * rkjz - muj_rkj3;
            double xykj = 3 * muj_rkj5 * rkjx * rkjy;
            double xzkj = 3 * muj_rkj5 * rkjx * rkjz;
            double yzkj = 3 * muj_rkj5 * rkjy * rkjz;

			a41 += xxij;
			a52 += yyij;
			a63 += zzij;
			a42 += xyij;
			a43 += xzij;
			a53 += yzij;

            sx += (xxkj - xxij) * vkjx + (xykj - xyij) * vkjy + (xzkj - xzij) * vkjz;
            sy += (xykj - xyij) * vkjx + (yykj - yyij) * vkjy + (yzkj - yzij) * vkjz;
            sz += (xzkj - xzij) * vkjx + (yzkj - yzij) * vkjy + (zzkj - zzij) * vkjz;
		}

		const double* stm = y + 6;
		double* dstm = yp + 6;

		for (int i = 0; i < 6; i++)
		{
			dstm[6 * i + 0] = stm[6 * i + 3];			
			dstm[6 * i + 1] = stm[6 * i + 4];			
			dstm[6 * i + 2] = stm[6 * i + 5];			
			dstm[6 * i + 3] = a41 * stm[6 * i] + a42 * stm[6 * i + 1] + a43 * stm[6 * i + 2];
			dstm[6 * i + 4] = a42 * stm[6 * i] + a52 * stm[6 * i + 1] + a53 * stm[6 * i + 2];
			dstm[6 * i + 5] = a43 * stm[6 * i] + a53 * stm[6 * i + 1] + a63 * stm[6 * i + 2];
		}

        const double* tau = y + 42;
		double* dtau = yp + 42;

        dtau[0] = tau[3];
        dtau[1] = tau[4];
        dtau[2] = tau[5];
        dtau[3] = a41 * tau[0] + a42 * tau[1] + a43 * tau[2] + sx;
        dtau[4] = a42 * tau[0] + a52 * tau[1] + a53 * tau[2] + sy;
        dtau[5] = a43 * tau[0] + a53 * tau[1] + a63 * tau[2] + sz;
	}
	
	// n-body equation of motion relative to reference body in ephemeris model
    // calculates state transition matrix and tau derivatives
	void n_body_df_stm_ephemeris_ref(double t, double y[], double yp[], void* params)
	{
		NBodyRefEphemerisParams* p = reinterpret_cast<NBodyRefEphemerisParams*>(params);

		auto [rk, vk] = p->ephemeris->get_position_velocity(p->reference_body, p->reference_time_scale * t);

		double rkix = y[0];
		double rkiy = y[1];
		double rkiz = y[2];

		double rki2 = rkix * rkix + rkiy * rkiy + rkiz * rkiz;
		double rki3 = sqrt(rki2) * rki2;
		double muk_rki3 = 1.0 / rki3;
		double muk_rki5 = muk_rki3 / rki2;

		yp[0] = y[3];
		yp[1] = y[4];
		yp[2] = y[5];
		yp[3] = -muk_rki3 * rkix;
		yp[4] = -muk_rki3 * rkiy;
		yp[5] = -muk_rki3 * rkiz;

		double a41 = 3 * muk_rki5 * rkix * rkix - muk_rki3;
		double a52 = 3 * muk_rki5 * rkiy * rkiy - muk_rki3;
		double a63 = 3 * muk_rki5 * rkiz * rkiz - muk_rki3;
		double a42 = 3 * muk_rki5 * rkix * rkiy;
		double a43 = 3 * muk_rki5 * rkix * rkiz;
		double a53 = 3 * muk_rki5 * rkiy * rkiz;

		double sx = 0;
		double sy = 0;
		double sz = 0;

		for (int j = 0; j < p->ephemeris->get_num_bodies(); j++)
		{
			double muj;
			Eigen::Vector3d rkj;
			Eigen::Vector3d vkj;

			if (j == p->reference_body)
			{
				rkj = -rk / p->reference_distance_scale;
				vkj = -vk / p->reference_velocity_scale;
				muj = p->ephemeris->get_muk() / p->reference_mu_scale;
			}
			else
			{
				std::tie(rkj, vkj) = p->ephemeris->get_position_velocity(j, p->reference_time_scale * t);
				rkj = (rkj - rk) / p->reference_distance_scale;
				vkj = (vkj - vk) / p->reference_velocity_scale;
				muj = p->ephemeris->get_mu(j) / p->reference_mu_scale;
			}

			double rkjx = rkj(0);
			double rkjy = rkj(1);
			double rkjz = rkj(2);

			double vkjx = vkj(0);
			double vkjy = vkj(1);
			double vkjz = vkj(2);

			double rkj2 = rkjx * rkjx + rkjy * rkjy + rkjz * rkjz;
			double rkj3 = sqrt(rkj2) * rkj2;
			double muj_rkj3 = muj / rkj3;
			double muj_rkj5 = muj_rkj3 / rkj2;

			double rijx = rkjx - y[0];
			double rijy = rkjy - y[1];
			double rijz = rkjz - y[2];

			double rij2 = rijx * rijx + rijy * rijy + rijz * rijz;
			double rij3 = sqrt(rij2) * rij2;
			double muj_rij3 = muj / rij3;
			double muj_rij5 = muj_rij3 / rij2;

			yp[3] += muj_rij3 * rijx - muj_rkj3 * rkjx;
			yp[4] += muj_rij3 * rijy - muj_rkj3 * rkjy;
			yp[5] += muj_rij3 * rijz - muj_rkj3 * rkjz;

			double xxij = 3 * muj_rij5 * rijx * rijx - muj_rij3;
			double yyij = 3 * muj_rij5 * rijy * rijy - muj_rij3;
			double zzij = 3 * muj_rij5 * rijz * rijz - muj_rij3;
			double xyij = 3 * muj_rij5 * rijx * rijy;
			double xzij = 3 * muj_rij5 * rijx * rijz;
			double yzij = 3 * muj_rij5 * rijy * rijz;

			double xxkj = 3 * muj_rkj5 * rkjx * rkjx - muj_rkj3;
			double yykj = 3 * muj_rkj5 * rkjy * rkjy - muj_rkj3;
			double zzkj = 3 * muj_rkj5 * rkjz * rkjz - muj_rkj3;
			double xykj = 3 * muj_rkj5 * rkjx * rkjy;
			double xzkj = 3 * muj_rkj5 * rkjx * rkjz;
			double yzkj = 3 * muj_rkj5 * rkjy * rkjz;

			a41 += xxij;
			a52 += yyij;
			a63 += zzij;
			a42 += xyij;
			a43 += xzij;
			a53 += yzij;

			sx += (xxkj - xxij) * vkjx + (xykj - xyij) * vkjy + (xzkj - xzij) * vkjz;
			sy += (xykj - xyij) * vkjx + (yykj - yyij) * vkjy + (yzkj - yzij) * vkjz;
			sz += (xzkj - xzij) * vkjx + (yzkj - yzij) * vkjy + (zzkj - zzij) * vkjz;
		}

		const double* stm = y + 6;
		double* dstm = yp + 6;

		for (int i = 0; i < 6; i++)
		{
			dstm[6 * i + 0] = stm[6 * i + 3];
			dstm[6 * i + 1] = stm[6 * i + 4];
			dstm[6 * i + 2] = stm[6 * i + 5];
			dstm[6 * i + 3] = a41 * stm[6 * i] + a42 * stm[6 * i + 1] + a43 * stm[6 * i + 2];
			dstm[6 * i + 4] = a42 * stm[6 * i] + a52 * stm[6 * i + 1] + a53 * stm[6 * i + 2];
			dstm[6 * i + 5] = a43 * stm[6 * i] + a53 * stm[6 * i + 1] + a63 * stm[6 * i + 2];
		}

		const double* tau = y + 42;
		double* dtau = yp + 42;

		dtau[0] = tau[3];
		dtau[1] = tau[4];
		dtau[2] = tau[5];
		dtau[3] = a41 * tau[0] + a42 * tau[1] + a43 * tau[2] + sx;
		dtau[4] = a42 * tau[0] + a52 * tau[1] + a53 * tau[2] + sy;
		dtau[5] = a43 * tau[0] + a53 * tau[1] + a63 * tau[2] + sz;
	}
}