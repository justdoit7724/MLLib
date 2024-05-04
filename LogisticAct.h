#pragma once
#include "Activation.h"
namespace ML
{
    class LogisticAct :
        public Activation
    {
    public:
        LogisticAct();
        Vector Calc(Vector z)const override;
        Matrix Diff(Vector z)const override;
    };
}

