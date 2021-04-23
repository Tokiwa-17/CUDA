#include <stdio.h>
#include <cuda_runtime.h>
#include "../include/matrixTile.cuh"
#define TILE_SIZE 16

__global__ void gpuMatrixMulTile(int* d_A, int* d_B, int* d_C, int m, int n, int k){
    
    __shared__ int A_tile[TILE_SIZE][TILE_SIZE];
    __shared__ int B_tile[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    // A是横着的条
    int aBegin = blockIdx.y * TILE_SIZE * n;
    int aEnd = aBegin + n - 1;
    int aStride = TILE_SIZE;
    // B是竖着的条
    int bBegin = TILE_SIZE * bx;
    int bStride = TILE_SIZE * k;

    int accu = 0;

    for(int i = aBegin, j = bBegin; i <= aEnd; i += aStride, j += bStride){
        //load share memory
        A_tile[ty][tx] = d_A[i + n * ty + tx];
        B_tile[tx][ty] = d_B[j + k * tx + ty];

        __syncthreads();

        for(int k = 0;k < TILE_SIZE; k++)
            accu += A_tile[ty][k] * B_tile[k][tx];
        
        __syncthreads();
    }
    int cIdx = k * TILE_SIZE * by + TILE_SIZE * bx;
    d_C[cIdx + k * ty + tx] = accu;
}
