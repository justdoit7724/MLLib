#pragma once
#include "MathHelp.h"

namespace ML {
	class Loss
	{
	public:

		/***
		Single data calculation
		***/
		virtual double Calculate(Vector yp, Vector yt, bool isLogit = false)=0;

		/***
		Single data calculation
		***/
		virtual Vector Gradient(Vector yp, Vector yt) = 0;
	};
}
