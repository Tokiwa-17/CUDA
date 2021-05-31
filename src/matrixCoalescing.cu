#include <stdio.h>
#include <cuda_runtime.h>
#include "../include/config.cuh"
#include "../include/matrixCoalescing.cuh"


__shared__ int A_tile[TILE_SIZE][TILE_SIZE];
__shared__ int B_tile[TILE_SIZE][TILE_SIZE];

int bx = blockIdx.x, by = blockIdx.y;
int tx = threadIdx.x, ty = threadIdx.y;

//illustrate :https://cnugteren.github.io/tutorial/pages/page4.html
// A是横着的条, aBegin和aEnd分别是Tile第一行的开始和结束, 每循环一次横着移动一个Tile
int aBegin = blockIdx.y * TILE_SIZE * n;
int aEnd = aBegin + n - 1;
int aStride = TILE_SIZE;

// B是竖着的条, bBegin指向Tile第一列的开始，每循环一次竖着移动一个Tile
int bBegin = TILE_SIZE * bx;
int bStride = TILE_SIZE * k;

int accu = 0;

// 计算C的一个Tile
for(int i = aBegin, j = bBegin; i <= aEnd; i += aStride, j += bStride){
    //load share memory
    //从Tile中取出一个点放到共享内存中
    A_tile[ty][tx] = d_A[i + n * ty + tx];
    B_tile[tx][ty] = d_B[i + n * ty + tx];

    __syncthreads();

    for(int k = 0;k < TILE_SIZE; k++)
        accu += A_tile[ty][k] * B_tile[k][tx];
    
    __syncthreads();
}
//A中横着的一行和B中竖着的一列累加完毕放到C中对应位置
int cIdx = k * TILE_SIZE * by + TILE_SIZE * bx;
d_C[cIdx + k * ty + tx] = accu;