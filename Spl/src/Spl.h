#pragma once


class Spl
{
public:
    Spl();
    Spl(double* t, double* c, size_t n, size_t k);

    double eval(double x);
private:
    const double* t_;
    const double tb_;
    const double te_;
    const double* c_;
    const size_t n_;
    const size_t k_;
};
