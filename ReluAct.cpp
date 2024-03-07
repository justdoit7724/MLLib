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

Matrix ReluAct::Diff(Vector z)
{
	int n = z.size();

	Matrix output(n, Vector(n,0));

	for (int i = 0; i < n; ++i)
	{
		output[i][i] = z[i] < 0 ? 0 : 1;
	}

	return output;
}
