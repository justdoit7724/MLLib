#include "pch.h"
#include "LinearAct.h"

using namespace ML;

Vector LinearAct::Calc(Vector z)
{
    return z;
}

Matrix LinearAct::Diff(Vector z)
{
    int n = z.size();
    return Identity(n);
}
