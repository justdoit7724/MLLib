#pragma once
#include <vector>

typedef std::vector<double> Vector1D;
typedef std::vector<std::vector<double>> Vector2D;

inline Vector1D Sigmoid(Vector1D& v)
{
	Vector1D ret;

	for (int i = 0; i < v.size(); ++i)
	{
		ret.push_back(1.0 / (1.0 + exp(-v[i])));
	}

	return ret;
}
