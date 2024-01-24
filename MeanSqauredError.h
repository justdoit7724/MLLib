#pragma once
#include "Loss.h"
namespace ML {
    class MeanSqauredError :
        public Loss
    {
    public:
        double Calculate(Vector1D& yp, Vector1D& yt, bool isLogit = false) override;
        void Gradient(Vector2D& x, Vector1D& yp, Vector1D& yt,Vector1D& gdw,double& gdb) override;
    };
}

