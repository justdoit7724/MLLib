#include "pch.h"
#include "LogisticAct.h"

using namespace ML;

Vector LogisticAct::Calc(Vector z)
{
	return Sigmoid(z);
}

Matrix LogisticAct::Diff(Vector z)
{
	int n = z.size();
	Matrix output(n, Vector(n, 0));

	for (int i = 0; i < n; ++i)
	{
		output[i][i] = exp(-z[i]) / pow(1 + exp(-z[i]), 2);
	}

	return output;
}
