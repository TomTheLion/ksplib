#pragma once


class Spl
{
public:

    Spl();
    Spl(double* t, double* c, int n, int k);
    Spl(double* tx, double* ty, double* c, int nx, int ny, int kx, int ky);

    double eval(double x, double y = 0.0);

private:
    double speval(double x, const double* t, const double* c, int n, int k);
    double bispeval(double x, double y);

    double* tx_;
    double* ty_;
    double* c_;
    int nx_;
    int ny_;
    int kx_;
    int ky_;
};
