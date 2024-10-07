#include "Spl.h"


Spl::Spl()
    : t_(nullptr), tb_(0.0), te_(0.0), c_(nullptr), n_(0), k_(0)
{

}

Spl::Spl(double* t, double* c, int n, int k)
	: t_(t), tb_(t[k]), te_(t[n - k - 1]), c_(c), n_(n), k_(k)
{

}

double Spl::eval(double x)
{
    double tb = t_[k_];
    double te = t_[n_ - k_ - 1];

    if (x < tb) x = tb;
    if (x > te) x = te;

    double d[3 + 1];

    int i = k_;
    while (t_[i + 1] < x)
        i++;

    for (size_t j = 0; j < k_ + 1; j++) {
        d[j] = c_[j + i - k_];
    }

    for (size_t r = 1; r < k_ + 1; r++) {
        for (size_t j = k_; j > r - 1; j--) {
            double alpha = (x - t_[j + i - k_]) / (t_[j + 1 + i - r] - t_[j + i - k_]);
            d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j];
        }
    }

    return d[k_];
}