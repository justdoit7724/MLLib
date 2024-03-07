#pragma once
#include "Loss.h"

namespace ML {
    class BinaryCrossEntropy :
        public Loss
    {
    public:
        double Calculate(Vector yp, Vector yt, bool isLogit = false) override;
        Vector Gradient(Vector yp, Vector yt) override;
    };
}

