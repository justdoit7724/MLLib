#include "pch.h"
#include "Activation.h"

using namespace ML;

Activation::Activation(ActKind kind)
	:m_kind(kind)
{
}

ActKind Activation::Kind() const
{
	return m_kind;
}
