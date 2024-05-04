#include "pch.h"
#include "FactoryLoss.h"
#include "BinaryCrossEntropy.h"
#include "CategoricalCrossEntropy.h"
#include "MeanSqauredError.h"
using namespace ML;

std::unique_ptr<Loss> FactoryLoss::Create(LossKind kind)
{
	switch (kind)
	{
	case ML::LossKind::BinaryCrossEntropy:
		return std::make_unique<BinaryCrossEntropy>();
		break;
	case ML::LossKind::CategoryCrossEntropy:
		return std::make_unique<CategoricalCrossEntropy>();
		break;
	case ML::LossKind::MeanSqure:
		return std::make_unique<MeanSqauredError>();
		break;
	default:
		break;
	}
}
