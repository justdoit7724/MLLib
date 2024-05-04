#pragma once
#include <memory>

namespace ML
{
	class Loss;
	enum class LossKind
	{
		BinaryCrossEntropy,
		CategoryCrossEntropy,
		MeanSqure
	};

	class FactoryLoss
	{
	public:
		static std::unique_ptr<Loss> Create(LossKind kind);
	};
}
