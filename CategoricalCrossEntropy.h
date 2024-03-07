#pragma once
#include "Loss.h"
namespace ML {
    class CategoricalCrossEntropy :
        public Loss
    {
        double Calculate(Vector yp, Vector yt, bool isLogit = false) override;
        Vector Gradient(Vector yp, Vector yt) override;
    };

}
