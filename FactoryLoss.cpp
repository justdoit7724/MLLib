#include "pch.h"
#include "FactoryLoss.h"
#include "BinaryCrossEntropy.h"
#include "CategoricalCrossEntropy.h"
#include "MeanSqauredError.h"
using namespace ML;

void FactoryLoss::Create(LossKind kind, Loss** out)
{
	switch (kind)
	{
	case ML::LossKind::BinaryCrossEntropy:
		*out = new BinaryCrossEntropy();
		break;
	case ML::LossKind::CategoryCrossEntropy:
		*out = new CategoricalCrossEntropy();
		break;
	case ML::LossKind::MeanSqure:
		*out = new MeanSqauredError();
		break;
	default:
		break;
	}
}
