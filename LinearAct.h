#pragma once
#include "Activation.h"

namespace ML
{
    class LinearAct :
        public Activation
    {
    public:

        Vector Calc(Vector z) override;

        Matrix Diff(Vector z, Matrix dz) override;


    };
}
