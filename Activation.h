#pragma once
#include "MathHelp.h"

namespace ML
{
	class Activation
	{
	public:
		virtual Vector Calc(Vector z) = 0;
		virtual Matrix Diff(Vector z) = 0;
	};
}

