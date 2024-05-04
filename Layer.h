#pragma once
#include <memory>
#include "MathHelp.h"
#include "FactoryAct.h"
#include "Activation.h"

namespace ML
{
	class Layer
	{
	public:
		Layer();
		void Initialize(int nInput, int nN, ActKind act);

		Vector Calc(Vector a, Vector& z);
		int GetInputSize()const;
		int GetNeurnSize()const;
		const Activation* GetAct()const;
		
		Matrix m_W;
		Vector m_B;
	private:
		int m_nInput;
		int m_nN;

		std::unique_ptr<Activation> m_act;
	};
}
