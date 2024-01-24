#pragma once
#include <vector>

#ifndef _MLLIB_MATH
#define _MLLIB_MATH

namespace ML {


	typedef std::vector<double> Vector;
	typedef std::vector<std::vector<double>> Matrix;

	void AddTo(Vector& a, const Vector& b);
	void SubTo(Vector& a, const Vector& b);
	double Dot(const Vector& a, const Vector& b);
	void AddTo(Vector& a, double v);
	void SubTo(Vector& a, double v);
	void MulTo(Vector& a, double v);
	void DivTo(Vector& a, double v);
	Vector Mul(const Vector& a, double v);
	
	void AddTo(Matrix& A, const Matrix& B);
	void SubTo(Matrix& A, const Matrix& B);
	Matrix Dot(const Matrix& A, const Matrix& B);
	Matrix Add(const Matrix& A, const Matrix& B);
	Matrix Sub(const Matrix& A, const Matrix& B);
	void AddTo(Matrix& A, double v);
	void SubTo(Matrix& A, double v);
	void MulTo(Matrix& A, double v);
	void DivTo(Matrix& A, double v);


	void Transpose(Matrix& A);

	double Sigmoid(double v);
	Vector Sigmoid(const Vector& v);
	

}
#endif
