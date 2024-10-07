// class JDate
// This class stores a Julian date and allows conversion to Gregorian and Kerbal dates

#pragma once

class Jdate
{
public:

	// Creates a Jdate object initialized at 1/1/1900 12:00:00
	Jdate();

    // Creates a Jdate object based on Julian date
    Jdate(double j);

    // Creates a Jdate object based on Gregorian date and time
	Jdate(int m, int d, int y, double t = 0.0);

    // Create a copy of another Jdate object
	Jdate(const Jdate& jdate);

	// Assign state from an existing Jdate object
	Jdate& operator = (const Jdate& jdate);

    // Add time to Jdate
	Jdate operator + (double t);

    // Subtract time from Jdate
	Jdate operator - (double t);

    // Subtract Jdate from Jdate
	double operator - (Jdate jd);

	// Destory Jdate object
	~Jdate();

    // Get methods
    double get_julian_date();
    double get_kerbal_time();
    int get_month();
    int get_day();
    int get_year();
    std::tuple<int, int, int> get_month_day_year();
    double get_time();

    // Set methods
    void set_julian_date(double j);
    void set_kerbal_time(double k);
    void set_month_day_year(int m, int d, int y, double t = 0.0);

private:

	// Julian date
	int j_;

    // time
    double t_;

    // Calendar month, day, year
    int m_;
    int d_;
    int y_;

    // Date conversion functions
    int julian_from_mdy(int m, int d, int y);
    std::tuple<int, int, int> mdy_from_julian(int j);
};

