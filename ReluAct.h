#pragma once
#include "Activation.h"
namespace ML
{
    class ReluAct :
        public Activation
    {
    public:
        ReluAct();
       Vector Calc(Vector z)const override;
       Matrix Diff(Vector z)const override;
    };
}

