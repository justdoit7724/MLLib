#include "pch.h"
#include "ReluAct.h"

using namespace ML;

Vector ReluAct::Calc(Vector z)
{
	Vector output(z.size());

	for (int i = 0; i < output.size(); ++i)
	{
		output[i] = z[i] < 0?0: z[i];

	}

	return output;
}

Matrix ReluAct::Diff(Vector z, Matrix dz)
{
	int na = dz.size();
	int nz = dz[0].size();

	Matrix output(na, Vector(nz));

	for (int y = 0; y < na; ++y)
	{
		for (int x = 0; x < na; ++x)
		{
			output[y][x] = dz[y][x] < 0? 0 : dz[y][x];
		}
	}

	return Matrix();
}
