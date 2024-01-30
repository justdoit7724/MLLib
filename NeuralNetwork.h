#pragma once
#include "_Math.h"
#include "FactoryAct.h"
#include "FactoryLoss.h"

namespace ML
{
	class Layer;
	class Loss;
	class NeuralNetwork
	{
	public:

		virtual ~NeuralNetwork();

		virtual void Compile(std::vector<std::pair<ActKind,int>> layers, LossKind loss, Matrix x, Vector y, bool isNormalize);

		Vector Predict(Vector x);
		void Release();

	protected:

		std::vector<Layer*> m_layers;
		Loss* m_loss;


		Matrix m_mx;
		Vector m_y;
		Vector m_mu;
		Vector m_sig;
	};
}
