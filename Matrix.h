#pragma once
#include <vector>

namespace ML {
	class Matrix
	{
	public:
		Matrix(int h, int w);
		void Reset(int h, int w);

		double Get(int y, int x) const;
		void Set(int y, int x, double v);
		int Width();
		int Height();
		Matrix& operator + (const Matrix& v);
		Matrix operator * (const Matrix& v);
		Matrix& operator - (const Matrix& v);
		Matrix& operator+(double v);
		Matrix& operator-(double v);
		Matrix& operator*(double v);
		Matrix& operator/(double v);

		Matrix& operator = (const Matrix& v);

	private:

		std::vector<std::vector<double>> m_data;
		int m_w;
		int m_h;
	};
}
