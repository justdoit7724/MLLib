#pragma once
#include "Activation.h"
namespace ML
{
    class LogisticAct :
        public Activation
    {
        Vector Calc(Vector z) override;
        Matrix Diff(Vector z) override;
    };
}

