#include "../include/cpuMatrixStrassen.cuh"
#pragma unroll

Matrix::Matrix(int n) : m_n(n){
    matrix = new int *[m_n];
    for(int i = 0;i < m_n;i++)
        matrix[i] = new int[m_n];
}

Matrix::Matrix(const Matrix& a) {
	m_n = a.m_n;
	matrix = new int* [m_n];
	for (int i = 0;i < m_n;i++)
		matrix[i] = new int[m_n];

	for (int i = 0;i < m_n;i++)
		for (int j = 0;j < m_n;j++)
			matrix[i][j] = a.matrix[i][j];
}


Matrix::Matrix(int *Ptr, int n){
    matrix = new int *[m_n];
    for(int i = 0;i < m_n;i++)
        matrix[i] = new int[m_n];

    for(int i = 0; i < m_n; i++)
        for(int j = 0; j < m_n; j++)
            matrix[i][j] = *(Ptr + n * i + j);
}

Matrix Matrix::operator = (const Matrix& a) {
	Matrix b(a.m_n);
	matrix = new int* [m_n];
	for (int i = 0;i < m_n;i++)
		matrix[i] = new int[m_n];

	for (int i = 0;i < m_n;i++)
		for (int j = 0;j < m_n;j++)
			matrix[i][j] = a.matrix[i][j];

	return b;
}

Matrix Matrix::operator + (const Matrix& a) {
	Matrix res(m_n);
	for (int i = 0;i < m_n;i++)
		for (int j = 0;j < m_n;j++)
			res.matrix[i][j] = matrix[i][j] + a.matrix[i][j];
	return res;
}
Matrix Matrix::operator - (const Matrix& a) {
	Matrix res(m_n);
	for (int i = 0;i < m_n;i++)
		for (int j = 0;j < m_n;j++)
			res.matrix[i][j] = this->matrix[i][j] - a.matrix[i][j];
	return res;
}

Matrix Matrix::operator * (const Matrix& a) {
	Matrix b(m_n);
	b.setZero();
	for (int i = 0;i < m_n;i++)
		for (int j = 0;j < m_n;j++)
			for (int k = 0;k < m_n;k++)
				b.matrix[i][j] += matrix[i][k] * a.matrix[k][j];
	return b;
}

bool Matrix::checkResult(int* ptr){
    double epsilon = 1.0E-8;
    for (int i = 0;i < m_n * m_n; i++){
        if (abs(ptr[i] - matrix[i]) > epsilon){
            printf("Results do not match!\n");
            printf("%d(hostRef[%d] )!= %d(gpuRef[%d])\n", ptr[i], i, matrix[i], i);
            return;
        }
    }
    printf("Check result success!\n");
}

cpuMatrixStrassen::cpuMatrixStrassen(int *ptrA, int *ptrB, int n){
    a = new Matrix(ptrA, n);
    b = new Matrix(ptrB, n);
}

void cpuMatrixStrassen::fill(Matrix& a, Matrix& b, int opt){
    switch (opt) {
    case 1:
        for (int i = 0;i < a.m_n;i++)
            for (int j = 0;j < a.m_n;j++)
                a.matrix[i][j] = b.matrix[i][j];
        break;

    case 2:
        for (int i = 0;i < a.m_n;i++)
            for (int j = 0;j < a.m_n;j++)
                a.matrix[i][j] = b.matrix[i][j + a.m_n];
        break;
    case 3:
        for (int i = 0;i < a.m_n;i++)
            for (int j = 0;j < a.m_n;j++)
                a.matrix[i][j] = b.matrix[i + a.m_n][j];
        break;
    case 4:
        for (int i = 0;i < a.m_n;i++)
            for (int j = 0;j < a.m_n;j++)
                a.matrix[i][j] = b.matrix[i + a.m_n][j + a.m_n];
        break;
    }
}

Matrix cpuMatrixStrassen::strassen(Matrix& a, Matrix& b){
    Matrix res(a.m_n);
    int n = a.m_n / 2;
	Matrix a11(n), a12(n), a21(n), a22(n), b11(n), b12(n), b21(n), b22(n),
		res11(n), res12(n), res21(n), res22(n), r(n), s(n), t(n), u(n);
	
    fill(a11, a, 1), fill(a12, a, 2), fill(a21, a, 3), fill(a22, a, 4);
	fill(b11, b, 1), fill(b12, b, 2), fill(b21, b, 3), fill(b22, b, 4);

    r = divide(a11, b11) + divide(a12, b21);
	s = divide(a11, b12) + divide(a12, b22);
	t = divide(a21, b11) + divide(a22, b21);
	u = divide(a21, b12) + divide(a22, b22);

    for (int i = 0;i < n;i++)
		for (int j = 0;j < n;j++) {
			res.matrix[i][j] = r.matrix[i][j];
			res.matrix[i][j + n] = s.matrix[i][j];
			res.matrix[i + n][j] = t.matrix[i][j];
			res.matrix[i + n][j + n] = u.matrix[i][j];
		}
	return res;
}
