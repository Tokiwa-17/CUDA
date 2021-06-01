#include <stdio.h>
#include <cuda_runtime.h>
#include "../include/config.cuh"
#include "../include/matrixPrefetch.cuh"

__global__ void gpuMatrixMulPrefetch(int* d_A, int* d_B, int* d_C, int m, int n, int k){
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    __shared__ int A_tile[TILE_SIZE * TILE_SIZE];
    __shared__ int A_tileNxt[TILE_SIZE * TILE_SIZE];
    
    //register for result of C at each thread
    int cval[TILE_SIZE];
    for(int i = 0;i < cval;i++) cval[i] = 0;

    int aBegin = n * TILE_SIZE * by;
    int aEnd = aBegin + n - 1;
    int aStride = TILE_SIZE;
    
    int bBegin = TILE_SIZE * VEC_SIZE * bx;
    int bStride = TILE_SIZE * k;

    int *cur = A_tile;
    int *nxt = A_tileNxt;
    for(int i = 0;i < TILE_SIZE / VEC_SIZE;i++)
        cur[(i * VEC_SIZE + ty) + TILE_SIZE * tx] = d_A[aBegin + n * (i * VEC_SIZE + ty) + tx];
    
    __syncthreads();

    for(int a = aBegin, b = bBegin; a <= aEnd; a += aStride, b += bStride){
        if(a + aStride <= aEnd){
            for(int i = 0;i < TILE_SIZE / VEC_SIZE; i++)
                nxt[(i * VEC_SIZE) + ty + TILE_SIZE * tx] = d_A[a + n * (i * VEC_SIZE + ty) + tx + aStride];
        }
        int *aptr = cur;
        int *bptr = &d_B[b + TILE_SIZE * ty + tx];

        for(int i = 0;i < TILE_SIZE;i++){
            int bval = *bptr;
            for(int j = 0;j < TILE_SIZE;j++)
                cval[j] += aptr[j] * bval;
            aptr += TILE_SIZE;
            bptr += k;
        }
        __syncthreads();

        int *tmp = cur;
        cur = nxt;
        nxt = tmp;
    }
    int cPos = k * TILE_SIZE * by + TILE_SIZE * VEC_SIZE * bx + TILE_SIZE * ty + tx;
    for(int i = 0;i < TILE_SIZE;i++){
        d_C[cPos] = cval[i];
        cPos += k;
    }
}
