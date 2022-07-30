#include <cmath>
#include <tuple>
#include "Jdate.h"

// Creates an uninitialized Jdate object
Jdate::Jdate()
{
    m_ = 1;
    d_ = 1;
    y_ = 1900;
    t_ = 0.0;
    j_ = julian_from_mdy(m_, d_, y_);

}

// Creates a Jdate object based on Julian date
Jdate::Jdate(double j)
{
    double a;
    t_ = std::modf(j, &a);
    j_ = static_cast<int>(a);
    std::tie(m_, d_, y_) = mdy_from_julian(j_);
}

// Creates a Jdate object based on Gregorian date and time
Jdate::Jdate(int m, int d, int y, double t)
    : m_(m), d_(d), y_(y), t_(t)
{
    j_ = julian_from_mdy(m, d, y);
}

// Create a copy of another Jdate object
Jdate::Jdate(const Jdate& jdate)
    : j_(jdate.j_), t_(jdate.t_), m_(jdate.m_), d_(jdate.d_), y_(jdate.y_)
{
    
}

// Assign state from an existing Jdate object
Jdate& 	Jdate::operator = (const Jdate& jdate)
{
    if (&jdate == this)
    {
        return *this;
    }

    j_ = jdate.j_;
    t_= jdate.t_;
    m_= jdate.m_;
    d_= jdate.d_;
    y_= jdate.y_;

    return *this;
}

// Add time to Jdate
Jdate Jdate::operator + (double t)
{
    Jdate jdate(this->get_julian_date() + t);

    return jdate;
}

// Subtract time from Jdate
Jdate Jdate::operator - (double t)
{
    Jdate jdate(this->get_julian_date() - t);

    return jdate;
}

// Subtract Jdate from Jdate
double Jdate::operator - (Jdate jd)
{
    return this->get_julian_date() - jd.get_julian_date();
}

// Destory Jdate object
Jdate::~Jdate()
{
    
}

// Get methods
double Jdate::get_julian_date()
{
    return j_ + t_;
}

double Jdate::get_kerbal_time()
{
    double k = (j_ + t_ - 2433647.5) * 86400.0;

    return k;
}

int Jdate::get_month()
{
    return m_;
}

int Jdate::get_day()
{
    return d_;
}

int Jdate::get_year()
{
    return y_;
}

std::tuple<int, int, int> Jdate::get_month_day_year()
{
    return { m_, d_, y_ };
}

double Jdate::get_time()
{
    return t_;
}

// Set methods
void Jdate::set_julian_date(double j)
{
    double a;
    t_ = std::modf(j, &a);
    j_ = static_cast<int>(a);
    std::tie(m_, d_, y_) = mdy_from_julian(j_);
}

void Jdate::set_kerbal_time(double k)
{
    double j = 2433647.5 + k / 86400.0;
    double a;
    t_ = std::modf(j, &a);
    j_ = static_cast<int>(a);
    std::tie(m_, d_, y_) = mdy_from_julian(j_);
}

void Jdate::set_month_day_year(int m, int d, int y, double t)
{
    m_ = m;
    d_ = d;
    y_ = y;
    t_ = t;
    j_ = julian_from_mdy(m_, d_, y_);
}

// Date conversion functions
int Jdate::julian_from_mdy(int m, int d, int y)
{
    int a = int((m - 14) / 12);
    int j = int((1461 * (y + 4800 + a)) / 4) + int((367 * (m - 2 - 12 * a)) / 12) - int((3 * int((y + 4900 + a) / 100)) / 4) + d - 32075;
    
    if (t_ >= 0.5)
        j--;

    return j;
}

std::tuple<int, int, int> Jdate::mdy_from_julian(int j)
{
    if (t_ >= 0.5)
        j++;

    int g = int(int((j - 4479.5) / 36524.25) * 0.75 + 0.5) - 37;
    int n = g + j;
    int y = int(n / 365.25) - 4712;
    int d = int(std::fmod(n - 59.25, 365.25));
    int m = (int((d + 0.5) / 30.6) + 2) % 12 + 1;
    d = int(fmod(d + 0.5, 30.6)) + 1;

    return { m, d, y };
}