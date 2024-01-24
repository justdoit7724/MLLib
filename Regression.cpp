#include "pch.h"
#include "Regression.h"
#include "Loss.h"

using namespace ML;


Regression::~Regression()
{
	delete m_loss;
}

bool Regression::Compile(Vector2D&  x, Vector1D& y, bool isNormalize)
{
	m_nVar = x[0].size();
	m_w.resize(m_nVar, 0);
	m_mx.resize(x.size(), Vector1D(m_nVar, 0));

	m_isNormalized = isNormalize;
	m_mx = m_isNormalized? ZNormalize(x,true):x;

	m_y = y;

	return false;
}

Vector1D Regression::Train(int iter, double alpha)
{
	Vector1D hCost;
	for (int i = 0; i < iter; ++i)
	{
		double cost = Cost(m_mx, m_y, m_w, m_b);
		hCost.push_back(cost);
		Vector1D gdw;
		double gdb;
		Gradient(m_mx, m_y, m_w, m_b,gdw,gdb);

		for (int i = 0; i < m_nVar; ++i)
		{
			m_w[i] -= gdw[i] * alpha;
		}
		m_b -= gdb * alpha;
	}

	return hCost;
}

Vector1D Regression::Predict(Vector2D&  x)
{
	Vector1D ret;
	Vector2D mx = m_isNormalized? ZNormalize(x): x;

	return Func(mx, m_w, m_b);
}

void Regression::SetWeights(Vector1D& w, double b)
{
	m_w = w;
	m_b = b;
}

void Regression::GetWeights(Vector1D& w, double& b)
{
	w = m_w;
	b = m_b;
}


Vector2D Regression::ZNormalize(Vector2D&  x, bool reCalc)
{
	int n = x[0].size();
	int m = x.size();

	if (reCalc)
	{
		m_mu.resize(n);
		m_sig.resize(n);
		for (int i = 0; i < n; ++i)
		{
			for (int j = 0; j < m; ++j)
			{
				m_mu[i] += x[j][i];
			}
			m_mu[i] /= m;
		}

		for (int i = 0; i < n; ++i)
		{
			for (int j = 0; j < m; ++j)
			{
				m_sig[i] += pow(x[j][i] - m_mu[i], 2);
			}
			m_sig[i] = sqrt(m_sig[i]);
		}
	}

	Vector2D ret;
	ret.resize(m, Vector1D(n, 0));

	for (int i = 0; i < m; ++i)
	{
		for(int j=0; j<n; ++j)
		{
			ret[i][j] = (x[i][j] - m_mu[j]) / m_sig[j];
		}
	}

	return ret;
}

void Regression::Gradient(Vector2D& x, Vector1D& y, Vector1D& w, double b, Vector1D& gdw, double& gdb)
{
	Vector1D yPred = Func(x, w, b);
	m_loss->Gradient(x, yPred, y, gdw,gdb);
}