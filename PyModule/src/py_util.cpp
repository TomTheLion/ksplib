#include "py_lib.h"


namespace py_util
{
	size_t array_size(py::object obj)
	{
		return obj.cast<py::array_t<double>>().size();
	}

	double* array_ptr(py::object obj)
	{
		return obj.cast<py::array_t<double>>().mutable_data();
	}

	void array_copy(py::array_t<double> src, py::array_t<double> dst)
	{
		dst.resize({ src.size() });
		std::copy(src.mutable_data(), src.mutable_data() + src.size(), dst.mutable_data());
	}

	void array_copy(std::vector<double> src, py::array_t<double> dst)
	{
		dst.resize({ src.size() });
		std::copy(src.data(), src.data() + src.size(), dst.mutable_data());
	}

	Eigen::Vector3d vector3d(py::object obj)
	{
		return Eigen::Vector3d(obj.cast<py::array_t<double>>().mutable_data());
	}

	Eigen::Vector3d vector3d(py::object obj, int n)
	{
		return Eigen::Vector3d(obj.cast<py::array_t<double>>().mutable_data() + n);
	}

	Spl spline(py::object obj)
	{
		py::tuple tup = obj;
		return Spl(array_ptr(tup[0]), array_ptr(tup[1]), array_size(tup[0]), tup[2].cast<size_t>());
	}

	double Spl::eval(double x) const
	{
		if (!t) return 0.0;

		double tb = t[k];
		double te = t[n - k - 1];
		x = x < tb ? tb : x > te ? te : x;

		size_t i = k;
		while (t[i + 1] < x)
			i++;

		double d[6];

		for (size_t j = 0; j < k + 1; j++)
			d[j] = c[j + i - k];

		for (size_t r = 1; r < k + 1; r++) {
			for (size_t j = k; j > r - 1; j--) {
				double alpha = (x - t[j + i - k]) / (t[j + 1 + i - r] - t[j + i - k]);
				d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j];
			}
		}

		return d[k];
	}
}