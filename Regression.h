#pragma once


namespace ML {
	class Loss;

	class Regression
	{
	public:
		virtual ~Regression();
		virtual bool Compile(Vector2D&  x, Vector1D& y, bool isNormalize=false);
		virtual Vector1D Train(int iter, double alpha);
		virtual Vector1D Predict(Vector2D&  x);
		void SetWeights(Vector1D& w, double b);
		void GetWeights(Vector1D& w, double& b);
		Vector2D ZNormalize(Vector2D&  x, bool reCalc = false);

		virtual Vector1D Func(Vector2D&  x, Vector1D& w, double b) = 0;
		virtual double Cost(Vector2D&  x, Vector1D& y, Vector1D& w, double b) = 0;
		void Gradient(Vector2D&  x, Vector1D& y, Vector1D& w, double b, Vector1D& gdw, double& gdb);

	protected:
		Loss* m_loss;
	private:
		Vector2D m_mx;
		Vector1D m_y;
		Vector1D m_w;
		double m_b;
		int m_nVar;


		bool m_isNormalized;
		Vector1D m_mu;
		Vector1D m_sig;

	};

}