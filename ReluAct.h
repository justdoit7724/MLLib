#pragma once
#include "Activation.h"
namespace ML
{
    class ReluAct :
        public Activation
    {
       Vector Calc(Vector z) override;
       Matrix Diff(Vector z, Matrix dz) override;
    };
}

