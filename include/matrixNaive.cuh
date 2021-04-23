#pragma once
#include <stdio.h>
#include <cuda_runtime.h>
#define BLOCK_SIZE 32

__global__ void gpuMatrixMul(int* d_A, int* d_B, int* d_C, int m, int n, int k);




