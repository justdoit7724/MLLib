#pragma once
#include "MathHelp.h"
#include "Loss.h"

namespace ML {

	class Regression
	{
	public:
		virtual ~Regression();
		virtual bool Compile(Matrix&  x, Vector& y, bool isNormalize=false);
		virtual Vector Train(int iter, double alpha);
		virtual Vector Predict(Matrix&  x);
		void SetWeights(Vector& w, double b);
		void GetWeights(Vector& w, double& b);
		Matrix ZNormalize(const Matrix& x);

		virtual Vector Func(Matrix&  x, Vector& w, double b) = 0;
		virtual double Cost(Matrix&  x, Vector& y, Vector& w, double b) = 0;
		void Gradient(Matrix&  x, Vector& y, Vector& w, double b, Vector& gdw, double& gdb);

	protected:
		Loss* m_loss;
	private:
		Matrix m_mx;
		Vector m_y;
		Vector m_w;
		double m_b;
		int m_nVar;


		bool m_isNormalized;
		Vector m_mu;
		Vector m_sig;

	};

}