#pragma once
#include "_Math.h"
#include "FactoryAct.h"

namespace ML
{
	class Activation;
	class Neuron;

	class Layer
	{
	public:
		Layer(int nInput, int nN, ActKind act);

		Vector Calc(Vector a);
		
	private:

		std::vector<Neuron*> m_neurons;
		Activation* m_act;
	};
}
