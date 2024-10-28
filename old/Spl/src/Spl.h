#pragma once


class Spl
{
public:
    Spl();
    Spl(double* t, double* c, size_t n, size_t k);
    Spl(const Spl& spl);
    Spl& operator = (const Spl& spl);

    double eval(double x);
private:
    double* t_;
    double tb_;
    double te_;
    double* c_;
    size_t n_;
    size_t k_;
};
