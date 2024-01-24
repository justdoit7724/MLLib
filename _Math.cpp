#include "pch.h"
#include "_Math.h"

void ML::AddTo(Vector& a, const Vector& b)
{
	assert(a.size() == b.size());

	for (int i = 0; i < a.size(); ++i)
	{
		a[i] += b[i];
	}
}

void ML::SubTo(Vector& a, const Vector& b)
{
	assert(a.size() == b.size());

	for (int i = 0; i < a.size(); ++i)
	{
		a[i] -= b[i];
	}

}

double ML::Dot(const Vector& a, const Vector& b)
{
	assert(a.size() == b.size());
	double ret = 0;
	for (int i = 0; a.size(); ++i)
	{
		ret += a[i] * b[i];
	}

	return ret;
}

void ML::AddTo(Vector& a, double v)
{
	for (int i = 0; i < a.size(); ++i)
	{
		a[i] += v;
	}
}

void ML::SubTo(Vector& a, double v)
{
	for (int i = 0; i < a.size(); ++i)
	{
		a[i] -= v;
	}
}

void ML::MulTo(Vector& a, double v)
{
	for (int i = 0; i < a.size(); ++i)
	{
		a[i] *= v;
	}
}

void ML::DivTo(Vector& a, double v)
{

	for (int i = 0; i < a.size(); ++i)
	{
		a[i] /= v;
	}
}

ML::Vector ML::Mul(const Vector& a, double v)
{
	Vector output = a;

	MulTo(output, v);
	return output;
}

void ML::AddTo(Matrix& A, const Matrix& B)
{
	assert(A.size() == B.size() && A[0].size() == B[0].size());

	for (int y = 0; y < A.size(); y++)
	{
		for (int x = 0; x < A[0].size(); ++x)
		{
			A[y][x] += B[y][x];
		}
	}
}

void ML::SubTo(Matrix& A, const Matrix& B)
{
	assert(A.size() == B.size() && A[0].size() == B[0].size());

	for (int y = 0; y < A.size(); y++)
	{
		for (int x = 0; x < A[0].size(); ++x)
		{
			A[y][x] -= B[y][x];
		}
	}
}

ML::Matrix ML::Dot(const Matrix& A, const Matrix& B)
{
	int hA = A.size();
	int wA = A[0].size();
	int hB = B.size();
	int wB = B[0].size();

	assert(wA == hB);

	Matrix output(hA, Vector(wB, 0));

	for (int x = 0; x < wB; x++) //2
	{
		for (int y = 0; y < hA; ++y) //1
		{
			double dotV = 0;
			for (int z = 0; z < wA; ++z)
			{
				dotV += A[y][z] * B[z][x];
			}
			output[y][x] = dotV;
		}
	}

	return output;
}


ML::Matrix ML::Add(const Matrix& A, const Matrix& B)
{
	assert(A.size() == B.size() && A[0].size() == B[0].size());

	Matrix output(A.size(), Vector(A[0].size()));

	for (int y = 0; y < A.size(); y++)
	{
		for (int x = 0; x < A[0].size(); ++x)
		{
			output[y][x] = A[y][x] + B[y][x];
		}
	}

	return output;
}


ML::Matrix ML::Sub(const Matrix& A, const Matrix& B)
{
	assert(A.size() == B.size() && A[0].size() == B[0].size());

	Matrix output(A.size(), Vector(A[0].size()));

	for (int y = 0; y < A.size(); y++)
	{
		for (int x = 0; x < A[0].size(); ++x)
		{
			output[y][x] = A[y][x] - B[y][x];
		}
	}

	return output;
}

void ML::AddTo(Matrix& A, double v)
{

	for (int y = 0; y < A.size(); y++)
	{
		for (int x = 0; x < A[0].size(); ++x)
		{
			A[y][x] += v;
		}
	}
}

void ML::SubTo(Matrix& A, double v)
{
	for (int y = 0; y < A.size(); y++)
	{
		for (int x = 0; x < A[0].size(); ++x)
		{
			A[y][x] -= v;
		}
	}
}

void ML::MulTo(Matrix& A, double v)
{
	for (int y = 0; y < A.size(); y++)
	{
		for (int x = 0; x < A[0].size(); ++x)
		{
			A[y][x] *= v;
		}
	}
}

void ML::DivTo(Matrix& A, double v)
{
	for (int y = 0; y < A.size(); y++)
	{
		for (int x = 0; x < A[0].size(); ++x)
		{
			A[y][x] /= v;
		}
	}
}


void ML::Transpose(Matrix& A)
{
}

double ML::Sigmoid(double v)
{
	return 1.0 / (1.0 + exp(-v));
}

ML::Vector ML::Sigmoid(const Vector& v)
{
	Vector output(v.size());

	for (int i = 0; i < v.size(); ++i)
	{
		output[i] = Sigmoid(v[i]);
	}

	return output;
}

