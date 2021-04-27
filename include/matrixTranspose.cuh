#pragma once
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cmath>
#include <cassert>

__global__ void intPtrToFloatPtr(int *in, float* out, unsigned int m, unsigned int n);

__global__ void floatPtrToIntPtr(float *in, int* out, unsigned int m, unsigned int n);

void matrixTranspose(int *A, int *B, int m, int n);
