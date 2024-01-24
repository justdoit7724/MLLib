#include "pch.h"
#include "MeanSqauredError.h"

using namespace ML;
double MeanSqauredError::Calculate(Vector1D& yp, Vector1D& yt, bool isLogit)
{
    int m = yp.size();

    double ret = 0;
    for (int i = 0; i < m; ++i)
    {
        ret += pow(yt[i] - yp[i], 2);
    }
    return ret/(m*2);
}


void MeanSqauredError::Gradient(Vector2D& x, Vector1D& yp, Vector1D& yt, Vector1D& gdw, double& gdb)
{
    gdw.clear();
    gdb = 0;

    int m = yp.size();
    int n = x[0].size();

    gdw.resize(m);
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            gdw[i] += (yp[j] - yt[j])*x[j][i];
        }
        gdw[i] /= m;
    }
    
    for (int i = 0; i < m; ++i)
    {
        gdb += yp[i] - yt[i];
    }
    gdb /= m;
}