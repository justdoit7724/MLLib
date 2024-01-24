#include "pch.h"
#include "LinearRegression.h"
#include "MeanSqauredError.h"

using namespace ML;
LinearRegression::LinearRegression()
{
	m_loss = new MeanSqauredError();
}

Vector1D LinearRegression::Func(Vector2D&  x, Vector1D& w, double b)
{
	int m = x.size();
	int n = x[0].size();

	Vector1D ret(m);
	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			ret[i] += x[i][j] * w[j];
		}
		ret[i] += b;
	}

	return ret;
}

double LinearRegression::Cost(Vector2D&  x, Vector1D& y, Vector1D& w, double b)
{
	Vector1D yPred = Func(x, w, b);

	return m_loss->Calculate(yPred, y, true);
}
