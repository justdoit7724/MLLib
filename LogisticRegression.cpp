#include "pch.h"
#include "LogisticRegression.h"
#include "BinaryCrossEntropy.h"

using namespace ML;
LogisticRegression::LogisticRegression()
{
	m_loss = new BinaryCrossEntropy();
}
Vector1D LogisticRegression::Func(Vector2D& x, Vector1D& w, double b)
{
	int m = x.size();
	int n = x[0].size();

	Vector1D ret(m);
	for (int i = 0; i < m; ++i)
	{
		double z = b;
		for (int j = 0; j < n; ++j)
		{
			z += x[i][j] * w[j];
		}

		ret[i] = 1.0 / (1.0 + exp(-z));
	}
	return ret;
}
double LogisticRegression::Cost(Vector2D& x, Vector1D& y, Vector1D& w, double b) {

	Vector1D yPred = Func(x, w, b);

	return m_loss->Calculate(yPred, y);
}
