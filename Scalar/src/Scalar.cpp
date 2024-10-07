#include "Scalar.h"


Scalar::Scalar()
{
	for (size_t i = 0; i < 8; i++)
		scalars_[i] = 1.0;
}

Scalar::Scalar(double time, double distance, double mass)
{
	double rate = 1.0 / time;
	double velocity = distance * rate;
	double acceleration = velocity * rate;

	scalars_[0] = time;
	scalars_[1] = rate;
	scalars_[2] = distance;
	scalars_[3] = velocity;
	scalars_[4] = acceleration;
	scalars_[5] = mass;
	scalars_[6] = mass * rate;
	scalars_[7] = mass * acceleration;
}


double Scalar::ndim(Quantity quantity, double value) const
{
	return value / scalars_[static_cast<int>(quantity)];
}

double Scalar::rdim(Quantity quantity, double value) const
{
	return value * scalars_[static_cast<int>(quantity)];
}

void Scalar::ndim(Quantity* quantities, double* values, size_t n) const
{
	for (size_t i = 0; i < n; i++)
	{
		values[i] /= scalars_[static_cast<int>(quantities[i])];
	}
}

void Scalar::rdim(Quantity* quantities, double* values, size_t n) const
{
	for (size_t i = 0; i < n; i++)
	{
		values[i] *= scalars_[static_cast<int>(quantities[i])];
	}
}

void Scalar::ndim(Quantity* quantities, double* values, size_t m, size_t n) const
{
	for (size_t i = 0; i < m; i++)
	{
		for (size_t j = 0; j < n; j++)
		{
			values[i * n + j] /= scalars_[static_cast<int>(quantities[j])];
		}
	}
}

void Scalar::rdim(Quantity* quantities, double* values, size_t m, size_t n) const
{
	for (size_t i = 0; i < m; i++)
	{
		for (size_t j = 0; j < n; j++)
		{
			values[i * n + j] *= scalars_[static_cast<int>(quantities[j])];
		}
	}
}