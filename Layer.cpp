#include "pch.h"
#include "Layer.h"
#include "Neuron.h"

using namespace ML;

Layer::Layer(int nInput, int nN, ActKind act)
{




	for (int i = 0; i < nN; ++i)
	{
		m_neurons.push_back(new Neuron(nInput));
	}

	FactoryAct::Create(act, &m_act);
}

Vector Layer::Calc(Vector a)
{
	Vector z;

	for (Neuron* n : m_neurons)
	{
		z.push_back(n->Calc(a));
	}

	return m_act->Calc(z);
}
