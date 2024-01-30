#pragma once
#include "Activation.h"

namespace ML
{
	enum class ActKind {
		Linear,
		Logistic,
		Relu,
		Softmax
	};

	class FactoryAct
	{
	public:
		static void Create(ActKind kind, Activation** out);
	};

}