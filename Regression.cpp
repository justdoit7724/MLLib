#include "pch.h"
#include "Regression.h"
#include "Loss.h"
#include "Normalizer.h"

using namespace ML;


Regression::~Regression()
{
	delete m_loss;
}

bool Regression::Compile(Matrix&  x, Vector& y, bool isNormalize)
{
	m_nVar = x[0].size();
	m_w.resize(m_nVar, 0);
	m_mx.resize(x.size(), Vector(m_nVar, 0));

	m_isNormalized = isNormalize;
	if (m_isNormalized)
	{
		Normalizer::ZNormalize(x, &m_mx, &m_mu, &m_sig);
	}
	else
		m_mx = x;

	m_y = y;

	return false;
}

Vector Regression::Train(int iter, double alpha)
{
	Vector hCost;
	for (int i = 0; i < iter; ++i)
	{
		double cost = Cost(m_mx, m_y, m_w, m_b);
		hCost.push_back(cost);
		Vector gdw;
		double gdb;
		Gradient(m_mx, m_y, m_w, m_b,gdw,gdb);

		SubTo(m_w, Mul(gdw, alpha));
		m_b -= gdb* alpha;
	}

	return hCost;
}

Vector Regression::Predict(Matrix&  x)
{
	Vector ret;
	Matrix mx = m_isNormalized? ZNormalize(x): x;

	return Func(mx, m_w, m_b);
}

void Regression::SetWeights(Vector& w, double b)
{
	m_w = w;
	m_b = b;
}

void Regression::GetWeights(Vector& w, double& b)
{
	w = m_w;
	b = m_b;
}

Matrix ML::Regression::ZNormalize(const Matrix& x)
{
	int n = x[0].size();
	int m = x.size();

	Matrix ret;
	ret.resize(m, Vector(n, 0));

	for (int i = 0; i < m; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			ret[i][j] = (x[i][j] - m_mu[j]) / m_sig[j];
		}
	}

	return ret;
}



void Regression::Gradient(Matrix& x, Vector& y, Vector& w, double b, Vector& gdw, double& gdb)
{
	Vector yPred = Func(x, w, b);
	m_loss->Gradient(x, yPred, y, gdw,gdb);
}