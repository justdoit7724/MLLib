#pragma once
#include "Regression.h"
namespace ML {
    class LinearRegression :
        public Regression
    {
    public:
        LinearRegression();

        Vector1D Func(Vector2D&  x, Vector1D& w, double b) override;
        double Cost(Vector2D&  x, Vector1D& y, Vector1D& w, double b) override;
    };
}

