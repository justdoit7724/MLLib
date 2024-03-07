#include "pch.h"
#include "NeuralNetwork.h"
#include "Normalizer.h"
#include "Layer.h"
#include "Loss.h"
#include "Activation.h"

using namespace ML;

ML::NeuralNetwork::NeuralNetwork()
	:m_normalized(false), m_loss(nullptr)
{
}

NeuralNetwork::~NeuralNetwork()
{
	Release();
}

void NeuralNetwork::Compile(std::vector<std::pair<ActKind, int>> layers, LossKind loss, Matrix x, Matrix y, bool isNormalize)
{
	Release();

	int na = x[0].size();

	FactoryLoss::Create(loss, &m_loss);

	m_layers.push_back(new Layer(na, layers.front().second,(ActKind)(layers.front().first)));
	for (int i=1; i< layers.size();++i)
	{
		ActKind kind = layers[i].first;
		int neuronNum = layers[i].second;

		m_layers.push_back(new Layer(layers[i - 1].second, neuronNum, kind));
	}

	m_y = y;
	if (m_normalized = isNormalize)
	{
		Normalizer::ZNormalize(x, &m_mx, &m_mu, &m_sig);
	}
	else
	{
		m_mx = x;

		m_mu.clear();
		m_sig.clear();
	}
}

Vector NeuralNetwork::Train(int iteration, double alpha)
{
	{

		Matrix pred = Predict(m_mx);
		double cost = 0;
		for (int i = 0; i < pred.size(); ++i)
		{
			cost += m_loss->Calculate(pred[i], m_y[i]);
		}
		cost /= pred.size();
	}

	Vector hist;
	for (int j = 0; j < iteration; ++j)
	{
		for (int i = 0; i < m_mx.size(); ++i)
		{
			Matrix A;
			Matrix Z;
			Vector tmp;
			A.push_back(m_layers[0]->Calc(m_mx[i], tmp));
			Z.push_back(tmp);

			for (int j = 1; j < m_layers.size(); ++j)
			{
				A.push_back(m_layers[j]->Calc(A[j - 1], tmp));
				Z.push_back(tmp);
			}

			//backpropagation
			// 
			//Cost diff
			m_prevDiff = ToMatrix(m_loss->Gradient(A.back(), m_y[i]));

			//layer diff
			for (int j = m_layers.size() - 1; j >= 0; --j)
			{
				Layer* layer = m_layers[j];

				
				Matrix dLZ = layer->Act()->Diff(Z[j]);

				//next diff mat before updating weights
				Matrix dLA = Transpose(Dot(Transpose(dLZ), layer->m_W));
				Matrix nextDiff = Dot(dLA, m_prevDiff);


				//make cur matrix
				Matrix dCZ = Dot(dLZ, m_prevDiff);

				//get gd
				if (j == 0)
				{
					for (int y = 0; y < layer->m_nInput; ++y)
					{
						for (int x = 0; x < layer->m_nN; ++x)
						{
							layer->m_W[x][y] -= dCZ[x][0] * m_mx[i][y] * alpha;
						}
					}
				}
				else
				{
					for (int y = 0; y < layer->m_nInput; ++y)
					{
						for (int x = 0; x < layer->m_nN; ++x)
						{
							layer->m_W[x][y] -= dCZ[x][0] * A[j - 1][y] * alpha;
						}


					}
				}
				for (int y = 0; y < layer->m_nN; ++y)
				{
					layer->m_B[y] -= dCZ[y][0] * alpha;
				}

				m_prevDiff= nextDiff;
			}
		}

		Matrix pred = Predict(m_mx);
		double cost = 0;
		for (int i = 0; i < pred.size(); ++i)
		{
			cost += m_loss->Calculate(pred[i], m_y[i]);
		}
		cost /= pred.size();
		hist.push_back(cost);
	}

	return hist;
}

Matrix NeuralNetwork::Predict(Matrix x)
{
	Matrix output;

	if (m_normalized)
	{
		for (int i = 0; i < x.size(); ++i)
		{
			x[i] = (x[i] - m_mu)/m_sig;
		}
	}

	for (int i=0; i< x.size(); ++i)
	{
		Vector v = x[i];

		for (Layer* l : m_layers)
		{
			Vector z;
			v = l->Calc(v, z);
		}

		output.push_back(v);
	}

	if (m_normalized)
	{
		for (int i = 0; i < output.size(); ++i)
		{
			output[i] = BinaryMul(output[i] , m_sig) + m_mu;
		}
	}

	return output;
}

void NeuralNetwork::Release()
{
	for (int i = 0; i < m_layers.size(); ++i)
	{
		delete m_layers[i];
	}
	m_layers.clear();

	m_sig.clear();
	m_mu.clear();
}

void ML::NeuralNetwork::SetWeights(int l, const Matrix& w, Vector b)
{
	m_layers[l]->m_W = w;
	m_layers[l]->m_B = b;
}

void ML::NeuralNetwork::GetWeights(int l, Matrix& w, Vector& b)
{
	b = m_layers[l]->m_B;
	w = m_layers[l]->m_W;
}
