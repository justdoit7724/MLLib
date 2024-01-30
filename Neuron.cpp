#include "pch.h"
#include "Neuron.h"

using namespace ML;

Neuron::Neuron(int na)
{
	m_w.resize(na, 0);
	m_b = 0;
}

double Neuron::Calc(Vector a)
{
	return Dot(a, m_w) + m_b;
}
