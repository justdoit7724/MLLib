#pragma once
#include "_Math.h"

namespace ML
{
	class Activation
	{
	public:
		virtual Vector Calc(Vector z) = 0;
		virtual Matrix Diff(Vector z, Matrix dz) = 0;
	};
}

