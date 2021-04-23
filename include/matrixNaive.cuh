#pragma once
#include <stdio.h>
#include <cuda_runtime.h>
#define BLOCK_SIZE 32

template <typename T>
void gpuMatrixMul(T* d_A, T* d_B, T* d_C, int m, int n, int k);
