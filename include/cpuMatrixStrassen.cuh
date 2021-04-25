#pragma once
#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
using namespace std;

class Matrix {
public:
	int m_n = 0;
	int** matrix;

	Matrix(int n);

    Matrix(int *Ptr, int n);

	Matrix(const Matrix& a);

	~Matrix();

	Matrix operator = (const Matrix& a);

	Matrix operator + (const Matrix& a);

	Matrix operator - (const Matrix& a);

	Matrix operator * (const Matrix& a);

    void checkResult(int *ptr);
};

class cpuMatrixStrassen{
public:
    Matrix *a, *b;
    cpuMatrixStrassen(int *ptrA, int *ptrB, int n);
    void fill(Matrix& a, Matrix& b, int opt);
    Matrix strassen(Matrix& a, Matrix &b);
}
