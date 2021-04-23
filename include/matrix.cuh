#pragma once
#include <stdio.h>
#include <cuda_runtime.h>
#include "freshman.h"
#define BLOCK_SIZE 32

__global__ void gpuMatrixMul(int* d_A, int* d_B, int* d_C, int m, int n, int k);

void cpuMatrixMul(int *h_A, int * h_B, int* h_C, int m, int n, int k);



