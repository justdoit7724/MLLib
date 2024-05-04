#pragma once
#include<memory>

namespace ML
{
	class Activation;
	enum class ActKind {
		Linear,
		Logistic,
		Relu,
		Softmax
	};

	class FactoryAct
	{
	public:
		static std::unique_ptr<Activation> Create(ActKind kind);
	};

}