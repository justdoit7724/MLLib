#pragma once
#include "Regression.h"
namespace ML
{
    class LogisticRegression :
        public Regression
    {
    public:
        LogisticRegression();


        Vector1D Func(Vector2D& x, Vector1D& w, double b) override;
        double Cost(Vector2D& x, Vector1D& y, Vector1D& w, double b) override;
    };
}
