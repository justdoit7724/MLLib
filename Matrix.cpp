#include "pch.h"
#include "Matrix.h"

using namespace ML;

Matrix::Matrix(int h, int w)
	:m_h(h), m_w(w)
{
	Reset(m_h, m_w);
}

void Matrix::Reset(int h, int w)
{
	m_h = h;
	m_w = w;
	m_data.resize(m_h);
	for (int y = 0; y < m_h; ++y)
	{
		m_data[y].resize(m_w,0);
	}
}

double Matrix::Get(int y, int x)const
{
	return m_data[y][x];
}

void Matrix::Set(int y, int x, double v)
{
	m_data[y][x] = v;
}

int Matrix::Width()
{
	return m_w;
}

int Matrix::Height()
{
	return m_h;
}

Matrix& Matrix::operator+(const Matrix& v)
{
	assert(m_h == v.m_h && m_w == v.m_w);

	for (int y = 0; y < m_h; ++y)
	{
		for (int x = 0; x < m_w; x++)
		{
			m_data[y][x] += v.Get(y, x);
		}
	}
	return *this;
}

Matrix Matrix::operator*(const Matrix& v)
{
	assert(m_h == v.m_w && m_w == v.m_h);

	Matrix output(m_h, v.m_w);

	for (int x = 0; x < v.m_w; x++) //2
	{
		for (int y = 0; y < m_h; ++y) //1
		{
			double dotV = 0;
			for (int z = 0; z < m_w; ++z)
			{
				 dotV += m_data[y][z] * v.m_data[z][x];
			}
			output.Set(y,x, dotV);
		}
	}

	return output;
}

Matrix& Matrix::operator-(const Matrix& v)
{
	assert(m_h == v.m_h && m_w == v.m_w);

	for (int y = 0; y < m_h; ++y)
	{
		for (int x = 0; x < m_w; x++)
		{
			m_data[y][x] -= v.Get(y, x);
		}
	}
	return *this;
}

Matrix& Matrix::operator+(double v)
{
	for (int y = 0; y < m_h; ++y)
	{
		for (int x = 0; x < m_w; x++)
		{
			m_data[y][x] += v;
		}
	}

	return *this;
}

Matrix& Matrix::operator-(double v)
{
	for (int y = 0; y < m_h; ++y)
	{
		for (int x = 0; x < m_w; x++)
		{
			m_data[y][x] -= v;
		}
	}
	return *this;
}

Matrix& Matrix::operator*(double v)
{
	for (int y = 0; y < m_h; ++y)
	{
		for (int x = 0; x < m_w; x++)
		{
			m_data[y][x] *= v;
		}
	}
	return *this;
}

Matrix& Matrix::operator/(double v)
{
	for (int y = 0; y < m_h; ++y)
	{
		for (int x = 0; x < m_w; x++)
		{
			m_data[y][x] /= v;
		}
	}
	return *this;
}

Matrix& Matrix::operator=(const Matrix& v)
{

	this->m_data = v.m_data;
	this->m_w = v.m_w;
	this->m_h = v.m_h;

	return *this;
}
