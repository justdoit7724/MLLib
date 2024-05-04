#include "pch.h"
#include "LinearAct.h"

using namespace ML;

ML::LinearAct::LinearAct()
    :Activation(ActKind::Linear)
{
}

Vector LinearAct::Calc(Vector z)const
{
    return z;
}

Matrix LinearAct::Diff(Vector z)const
{
    int n = z.size();
    return Identity(n);
}
