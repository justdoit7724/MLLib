#pragma once
#include "Loss.h"
namespace ML {
    class CategoricalCrossEntropy :
        public Loss
    {
        double Calculate(Vector& yp, Vector& yt, bool isLogit = false) override;
        void Gradient(Matrix& x, Vector& yp, Vector& yt, Vector& gdw, double& gdb) override;
    };

}
