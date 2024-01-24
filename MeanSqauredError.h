#pragma once
#include "Loss.h"
namespace ML {
    class MeanSqauredError :
        public Loss
    {
    public:
        double Calculate(Vector& yp, Vector& yt, bool isLogit = false) override;
        void Gradient(Matrix& x, Vector& yp, Vector& yt,Vector& gdw,double& gdb) override;
    };
}

