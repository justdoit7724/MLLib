#include "pch.h"
#include "CategoricalCrossEntropy.h"

using namespace ML;
double CategoricalCrossEntropy::Calculate(Vector1D& yp, Vector1D& yt, bool isLogit)
{
    return 0.0;
}

void CategoricalCrossEntropy::Gradient(Vector2D& x, Vector1D& yp, Vector1D& yt, Vector1D& gdw, double& gdb)
{

}
