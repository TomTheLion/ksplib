#pragma once


class Spl
{
public:

    Spl();
    Spl(double* t, double* c, int n, int k);

    double eval(double x);

private:

    double d_[4];
    const double* t_;
    const double tb_;
    const double te_;
    const double* c_;
    const size_t n_;
    const size_t k_;
};
