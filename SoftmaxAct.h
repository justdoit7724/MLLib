#pragma once
#include "Activation.h"
namespace ML
{
    class SoftmaxAct :
        public Activation
    {
    public:
        SoftmaxAct();

        Vector Calc(Vector z)const override;
        Matrix Diff(Vector z)const override;
    };
}

