#include "pch.h"
#include "Layer.h"

using namespace ML;

Layer::Layer(int nInput, int nN, ActKind act)
	:m_nInput(nInput),m_nN(nN)
{
	m_W.resize(nN, Vector(nInput, 1));
	m_B.resize(nN, 0);

	FactoryAct::Create(act, &m_act);
}

Vector Layer::Calc(Vector a, Vector& z)
{
	z = Transpose(Dot(m_W, ToMatrix(a)))[0] + m_B;

	return m_act->Calc(z);
}

Activation* ML::Layer::Act()
{
	return m_act;
}
