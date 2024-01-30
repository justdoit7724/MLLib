#include "pch.h"
#include "LogisticAct.h"
#include "_Math.h"

using namespace ML;

Vector LogisticAct::Calc(Vector z)
{
	return Sigmoid(z);
}

Matrix LogisticAct::Diff(Vector z, Matrix dz)
{
	int na = dz.size();
	int nz = dz[0].size();

	Matrix output(na, Vector(nz, 0));

	for (int x = 0; x < nz; ++x)
	{
		double db = exp(-z[x]) / pow(1 + exp(-z[x]), 2);
		for (int y = 0; y < na; ++y)
		{
			output[y][x] = db * dz[y][x];
		}
	}

	return output;
}
