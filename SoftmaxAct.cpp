#include "pch.h"
#include "SoftmaxAct.h"

using namespace ML;

Vector SoftmaxAct::Calc(Vector z)
{
	Vector output = Exp(z);
	double sum = std::accumulate(z.begin(), z.end(),0);
	DivTo(output, sum);

	return output;
}

Matrix SoftmaxAct::Diff(Vector z, Matrix dz)
{
	int na = dz.size();
	int nz = dz[0].size();

	Matrix output(na, Vector(nz));

	Vector expZ = Exp(z);
	double sum = std::accumulate(expZ.begin(), expZ.end(), 0);

	for (int y = 0; y < na; ++y)
	{
		double mulSum = Dot(expZ, dz[y]);

		for (int x = 0; x < nz; ++x)
		{
			output[y][x] = expZ[x] * dz[y][x] * sum - mulSum * expZ[x];
		}
	}

	DivTo(output, sum * sum);

	return output;
}
