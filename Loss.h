#pragma once
namespace ML {
	class Loss
	{
	public:

		virtual double Calculate(Vector1D& yp, Vector1D& yt, bool isLogit = false) = 0;
		virtual void Gradient(Vector2D&  x, Vector1D& yp, Vector1D& yt, Vector1D& gdw, double& gdb) = 0;
	};
}
