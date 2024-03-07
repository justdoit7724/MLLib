#include "pch.h"
#include "BinaryCrossEntropy.h"

using namespace ML;

double BinaryCrossEntropy::Calculate(Vector yp, Vector yt, bool isLogit)
{
	auto& my = yp;

	if (isLogit)
	{
		my = Sigmoid(yp);
	}

	return -yt[0] * log(my[0]) - (1 - yt[0]) * log(1 - my[0]);
}

Vector BinaryCrossEntropy::Gradient(Vector yp, Vector yt)
{
	Vector gd(1, 0);
	
	gd[0] += yp[0] - yt[0];

	return gd;
}
