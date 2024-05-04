#pragma once
#include "MathHelp.h"
#include "FactoryAct.h"

namespace ML
{
	class Activation
	{
	public:
		Activation() {}
		Activation(ActKind kind);
		~Activation() {}

		virtual Vector Calc(Vector z)const = 0;
		virtual Matrix Diff(Vector z)const = 0;

		ActKind Kind()const;

	private:
		ActKind m_kind;

	};
}

