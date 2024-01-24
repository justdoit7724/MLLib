#include "pch.h"
#include "BinaryCrossEntropy.h"

using namespace ML;

double BinaryCrossEntropy::Calculate(Vector1D& yp, Vector1D& yt, bool isLogit)
{
	auto& my = yp;

	if (isLogit)
	{
		my = Sigmoid(yp);
	}

	double ret = 0;
	int m = yp.size();
	for (int i = 0; i < m; ++i)
	{
		ret += -yt[i] * log(my[i]) - (1 - yt[i]) * log(1 - my[i]);
	}

	return ret/m;
}

void BinaryCrossEntropy::Gradient(Vector2D& x, Vector1D& yp, Vector1D& yt,Vector1D& gdw, double& gdb)
{
	gdw.clear();
	gdb = 0;

	int m = yp.size();
	int n = x[0].size();

	gdw.resize(n);
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < m; ++j)
		{
			gdw[i] += (yp[j] - yt[j]) * x[j][i];
		}
		gdw[i] /= m;
	}
	for (int i = 0; i < m; ++i)
	{
		gdb += yp[i] - yt[i];
	}
	gdb /= m;
}
