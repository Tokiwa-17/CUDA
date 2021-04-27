#pragma once
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cmath>
#include <cassert>

void gpuMatrixCublas(int* A, int* B, int* C, int lda, int ldb, int ldc,
    int m, int n, int k, float alpha, float beta);
