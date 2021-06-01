#pragma once
#include <stdio.h>
#include <cuda_runtime.h>


__global__ void gpuMatrixMulBankConflict(int* d_A, int* d_B, int* d_C, int m, int n, int k);
