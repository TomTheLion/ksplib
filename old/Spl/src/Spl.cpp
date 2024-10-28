#include "Spl.h"


Spl::Spl()
    : t_(nullptr), tb_(0.0), te_(0.0), c_(nullptr), n_(0), k_(0)
{

}

Spl::Spl(double* t, double* c, size_t n, size_t k)
	: t_(t), tb_(t[k]), te_(t[n - k - 1]), c_(c), n_(n), k_(k)
{

}

Spl::Spl(const Spl& spl)
{
    t_ = spl.t_;
    tb_ = spl.tb_;
    te_ = spl.te_;
    c_ = spl.c_;
    n_ = spl.n_;
    k_ = spl.k_;
}

Spl& Spl::operator = (const Spl& spl)
{
	if (&spl == this)
	{
		return *this;
	}

    t_ = spl.t_;
    tb_ = spl.tb_;
    te_ = spl.te_;
    c_ = spl.c_;
    n_ = spl.n_;
    k_ = spl.k_;

     return *this;
}


double Spl::eval(double x)
{
    if (!t_) return 0.0;

    x = x < tb_ ? tb_ : x > te_ ? te_ : x;

    size_t i = k_;
    while (t_[i + 1] < x)
        i++;

    double d[6] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

    for (size_t j = 0; j < k_ + 1; j++)
        d[j] = c_[j + i - k_];

    for (size_t r = 1; r < k_ + 1; r++) {
        for (size_t j = k_; j > r - 1; j--) {
            double alpha = (x - t_[j + i - k_]) / (t_[j + 1 + i - r] - t_[j + i - k_]);
            d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j];
        }
    }

    return d[k_];
}