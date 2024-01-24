#include "pch.h"
#include "CategoricalCrossEntropy.h"

using namespace ML;
double CategoricalCrossEntropy::Calculate(Vector& yp, Vector& yt, bool isLogit)
{
    return 0.0;
}

void CategoricalCrossEntropy::Gradient(Matrix& x, Vector& yp, Vector& yt, Vector& gdw, double& gdb)
{

}
