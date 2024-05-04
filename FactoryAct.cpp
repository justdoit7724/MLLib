#include "pch.h"
#include "FactoryAct.h"
#include "LinearAct.h"
#include "LogisticAct.h"
#include "SoftmaxAct.h"
#include "ReluAct.h"

using namespace ML;

std::unique_ptr<Activation> FactoryAct::Create(ActKind kind)
{
	switch (kind)
	{
	case ActKind::Linear:
		return std::make_unique<LinearAct>();
	case ActKind::Logistic:
		return std::make_unique<LogisticAct>();
	case ActKind::Relu:
		return std::make_unique<ReluAct>();
	case ActKind::Softmax:
		return std::make_unique<SoftmaxAct>();
	default:
		return nullptr;
	}
}
