#include "../include/cpuMatrixStrassen.cuh"

Matrix::Matrix(int n) : m_n(n) {
    matrix = new int* [m_n];
    for (int i = 0;i < m_n;i++)
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


Matrix::Matrix(int* Ptr, int n) {
    m_n = n;
    matrix = new int* [m_n];
    for (int i = 0;i < m_n;i++)
        matrix[i] = new int[m_n];

    for(int i = 0; i < m_n; i++){
        for(int j = 0; j < m_n; j++)
            matrix[i][j] = *(Ptr + i * m_n + j);
    }
}

Matrix::~Matrix() {
    for (int i = 0;i < m_n;i++)
        if (matrix[i]) {
            delete[] matrix[i];
            matrix[i] = nullptr;
        }
    for (int i = 0;i < m_n;i++)
        if (matrix[i] != nullptr) {
            delete[] matrix[i];
            matrix[i] = nullptr;
        }
    //if (matrix) delete[] matrix;
}

void Matrix::setZero() {
    for (int i = 0; i < m_n; i++) {
        int* ptr = matrix[i];
        for (int j = 0;j < m_n; j++)
            *(ptr++) = 0;
    }

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

void Matrix::checkResult(int* ptr) {
    double epsilon = 1.0E-8;
    for (int i = 0;i < m_n; i++) {
        int *imal = matrix[i];
        int *iptr = ptr + i * m_n;
        for(int j = 0; j < m_n; j++){
            if (abs(iptr[j] - imal[j]) > epsilon) {
                printf("Results do not match!\n");
                printf("%d(hostRef[%d] )!= %d(gpuRef[%d])\n", ptr[i], i, imal[i], i);
                return;
            }
        }
    }
    printf("Check result success!\n");
}

void Matrix::printMatrix() {
    int* ic;
    printf("Matrix<%d,%d>:\n", m_n, m_n);
    for (int i = 0;i < m_n;i++) {
        ic = matrix[i];
        for (int j = 0;j < m_n;j++) {
            printf("%6d ", ic[j]);
        }
        printf("\n");
    }
}

void fill(Matrix& a, Matrix& b, int opt) {// opt 1: 11 2: 12 3: 21 4:22
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

Matrix strassen(Matrix& a, Matrix& b) {
    Matrix res(a.m_n);
    res.setZero();

    if (a.m_n <= 64) {
        res = a * b;
        return res;
    }

    int n = a.m_n / 2;
    Matrix a11(n), a12(n), a21(n), a22(n), b11(n), b12(n), b21(n), b22(n),
        res11(n), res12(n), res21(n), res22(n), r(n), s(n), t(n), u(n);
    fill(a11, a, 1), fill(a12, a, 2), fill(a21, a, 3), fill(a22, a, 4);
    fill(b11, b, 1), fill(b12, b, 2), fill(b21, b, 3), fill(b22, b, 4);

    r = strassen(a11, b11) + strassen(a12, b21);
    s = strassen(a11, b12) + strassen(a12, b22);
    t = strassen(a21, b11) + strassen(a22, b21);
    u = strassen(a21, b12) + strassen(a22, b22);

    for (int i = 0;i < n;i++)
        for (int j = 0;j < n;j++) {
            res.matrix[i][j] = r.matrix[i][j];
            res.matrix[i][j + n] = s.matrix[i][j];
            res.matrix[i + n][j] = t.matrix[i][j];
            res.matrix[i + n][j + n] = u.matrix[i][j];
        }
    return res;
}
