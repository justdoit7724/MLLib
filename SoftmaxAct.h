#pragma once
#include "Activation.h"
namespace ML
{
    class SoftmaxAct :
        public Activation
    {
        Vector Calc(Vector z) override;
        Matrix Diff(Vector z, Matrix dz) override;
    };
}

