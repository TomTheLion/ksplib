#pragma once


class Scalar
{
public:
	Scalar();
	Scalar(double time, double distance, double mass);

	enum class Quantity
	{
		TIME = 0,
		RATE = 1,
		DISTANCE = 2,
		VELOCITY = 3,
		ACCELERATION = 4,
		MASS = 5,
		MASS_RATE = 6,
		FORCE = 7
	};

	double ndim(Quantity quantity, double value) const;
	double rdim(Quantity quantity, double value) const;
	void ndim(Quantity* quantities, double* values, size_t n) const;
	void rdim(Quantity* quantities, double* values, size_t n) const;
	void ndim(Quantity* quantities, double* values, size_t m, size_t n) const;
	void rdim(Quantity* quantities, double* values, size_t m, size_t n) const;
private:
	double scalars_[8];
};