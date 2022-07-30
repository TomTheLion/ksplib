#include "Spl.h"
#include <vector>


Spl::Spl(double* t, double* c, int n, int k)
	: tx_(t), ty_(nullptr), c_(c), nx_(n), ny_(0), kx_(k), ky_(0)
{

}


Spl::Spl(double* tx, double* ty, double* c, int nx, int ny, int kx, int ky)
	: tx_(tx), ty_(ty), c_(c), nx_(nx), ny_(ny), kx_(kx), ky_(ky)
{

}


double Spl::eval(double x, double y)
{
    if (ty_)
    {
        return bispeval(x, y);
    }
    else
    {
        return speval(x, tx_, c_, nx_, kx_);
    }
}


double Spl::speval(double x, const double* t, const double* c, int n, int k)
{
    double tb = t[k];
    double te = t[n - k - 1];

    if (x < tb) x = tb;
    if (x > te) x = te;

    std::vector<double> d(k + 1);

    int i = k;
    while (t[i + 1] < x)
        i++;

    for (size_t j = 0; j < k + 1; j++) {
        d[j] = c[j + i - k];
    }

    for (size_t r = 1; r < k + 1; r++) {
        for (size_t j = k; j > r - 1; j--) {
            double alpha = (x - t[j + i - k]) / (t[j + 1 + i - r] - t[j + i - k]);
            d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j];
        }
    }

    return d[k];
}


double Spl::bispeval(double x, double y)
{
    int nx = nx_ - kx_ - 1;
    int ny = ny_ - ky_ - 1;

    std::vector<double> cx(nx);

    for (size_t j = 0; j < nx; j++) {
        cx[j] = speval(y, ty_, &c_[ny * j], ny, ky_);
    }

    return speval(x, tx_, cx.data(), nx, kx_);
}
