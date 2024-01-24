#pragma once
#include "_Math.h"

namespace ML {
	class Loss
	{
	public:

		virtual double Calculate(Vector& yp, Vector& yt, bool isLogit = false) = 0;
		virtual void Gradient(Matrix&  x, Vector& yp, Vector& yt, Vector& gdw, double& gdb) = 0;
	};
}
