#pragma once
#include "MathHelp.h"
#include "FactoryAct.h"
#include "FactoryLoss.h"

namespace ML
{
	class Layer;
	class Loss;
	class NeuralNetwork
	{
	public:

		NeuralNetwork();
		virtual ~NeuralNetwork();

		virtual void Compile(std::vector<std::pair<ActKind,int>> layers, LossKind loss, Matrix x, Matrix y, bool isNormalize);
		virtual Vector Train(int iteration, double alpha);

		Matrix Predict(Matrix x);
		void Release();

		void SetWeights(int l, const Matrix& w, Vector b);
		void GetWeights(int l, Matrix& w, Vector& b);

	protected:

		std::vector<Layer*> m_layers;
		Loss* m_loss;

		bool m_normalized;
		Matrix m_mx;
		Matrix m_y;
		Vector m_mu;
		Vector m_sig;

		Matrix m_prevDiff;
	};
}
