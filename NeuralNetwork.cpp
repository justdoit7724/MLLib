#include "pch.h"
#include "NeuralNetwork.h"
#include "Normalizer.h"
#include "Layer.h"

using namespace ML;

ML::NeuralNetwork::~NeuralNetwork()
{
	Release();
}

void NeuralNetwork::Compile(std::vector<std::pair<ActKind, int>> layers, LossKind loss, Matrix x, Vector y, bool isNormalize)
{
	Release();

	int na = x[0].size();

	FactoryLoss::Create(loss, &m_loss);

	m_layers.push_back(new Layer(na, layers.front().second,(ActKind)(layers.front().first)));
	for (int i=1; i< layers.size();++i)
	{
		ActKind kind = layers[i].first;
		int neuronNum = layers[i].second;

		m_layers.push_back(new Layer(layers[i - 1].second, neuronNum, kind));
	}

	m_y = y;
	if (isNormalize)
	{
		Normalizer::ZNormalize(x, &m_mx, &m_mu, &m_sig);
	}
	else
	{
		m_mx = x;
	}
}

Vector NeuralNetwork::Predict(Vector x)
{
	Vector a = x;

	for (Layer* l : m_layers)
	{
		a = l->Calc(a);
	}

	return a;
}

void ML::NeuralNetwork::Release()
{
	for (int i = 0; i < m_layers.size(); ++i)
	{
		delete m_layers[i];
	}
	m_layers.clear();

	m_sig.clear();
	m_mu.clear();
}
