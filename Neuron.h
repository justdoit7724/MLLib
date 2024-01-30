#pragma once
#include "_Math.h"

namespace ML
{
	class Neuron
	{
	private:

		friend class Layer;
		Neuron(int na);

		double Calc(Vector a);

		Vector m_w;
		double m_b;
	};
}
