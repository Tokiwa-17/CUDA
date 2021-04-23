#pragma once
#include <stdio.h>
#include <cuda_runtime.h>
#define TILE_SIZE

__global__ void gpuMatrixMulTile(int* d_A, int* d_B, int* d_C, int m, int n, int k);
