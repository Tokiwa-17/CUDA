#pragma once
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cmath>
#include <cassert>

__global__ void intPtrToFloatPtr(int *in, float* out, unsigned int m, unsigned int n);

__global__ void floatPtrToIntPtr(float *in, int* out, unsigned int m, unsigned int n);

void gpuMatrixCublas(int* A, int* B, int* C, int lda, int ldb, int ldc,
    int m, int n, int k, float alpha, float beta);
