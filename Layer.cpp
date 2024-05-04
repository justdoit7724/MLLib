#include "pch.h"
#include "Layer.h"
#include "Activation.h"

using namespace ML;


ML::Layer::Layer()
	:m_nInput(0), m_nN(0)
{
}

void ML::Layer::Initialize(int nInput, int nN, ActKind act)
{
	m_nInput = nInput;
	m_nN = nN;

	m_W.resize(nN, Vector(nInput));
	for (int i = 0; i < nN; ++i)
	{
		for (int j = 0; j < nInput; ++j)
		{
			m_W[i][j] = 2 * (rand() / (float)RAND_MAX - 0.5);
		}
	}
	m_B.resize(nN);
	for (int i = 0; i < nN; ++i)
	{
		m_B[i] = 2 * (rand() / (float)RAND_MAX - 0.5);
	}

	m_act=FactoryAct::Create(act);
}

Vector Layer::Calc(Vector a, Vector& z)
{
	z = Transpose(Dot(m_W, ToMatrix(a)))[0] + m_B;

	return m_act->Calc(z);
}


int ML::Layer::GetInputSize() const
{
	return m_nInput;
}

int ML::Layer::GetNeurnSize() const
{
	return m_nN;
}


const Activation* ML::Layer::GetAct() const
{
	return m_act.get();
}
