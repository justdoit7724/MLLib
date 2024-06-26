#include "pch.h"
#include "BinaryCrossEntropy.h"

using namespace ML;

ML::BinaryCrossEntropy::BinaryCrossEntropy()
	:Loss(LossKind::BinaryCrossEntropy)
{
}

double BinaryCrossEntropy::Calculate(Vector yp, Vector yt, bool isLogit)
{
	int n = yp.size();

	auto& my = yp;

	if (isLogit)
	{
		my = Sigmoid(yp);
	}

	auto eps_my =std::max(std::min(my[0], 1.0 - DX_EPSILON), DX_EPSILON);

	double cost = 0;
	for (int i = 0; i < n; ++i)
	{
		cost += -yt[i] * log(eps_my) - (1 - yt[i]) * log(1 - eps_my);;
	}

	return cost;
}

Vector BinaryCrossEntropy::Gradient(Vector yp, Vector yt)
{
	int n = yp.size();

	Vector gd(n, 0);
	for(int i=0; i<n;++i)
		gd[i] += yp[i] - yt[i];

	return gd;
}
