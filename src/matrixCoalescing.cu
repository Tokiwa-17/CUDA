#include <stdio.h>
#include <cuda_runtime.h>
#include "../include/config.cuh"
#include "../include/matrixCoalescing.cuh"



//矩阵的大小设置成TILE_SIZE 的倍数
__global__ void gpuMatrixMulCoalescing(int* d_A, int* d_B, int* d_C, int m, int n, int k){
    
    __shared__ int A_tile[TILE_SIZE][TILE_SIZE];
    __shared__ int B_tile[TILE_SIZE][TILE_SIZE];

    //一个线程块计算矩阵C的一个tile
    //线程块中的每一个线程计算tile中的一个元素
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    //illustrate :https://cnugteren.github.io/tutorial/pages/page4.html
    // A是横着的条, aBegin和aEnd分别是Tile第一行的开始和结束, 每循环一次横着移动一个Tile
    int aBegin = blockIdx.y * TILE_SIZE * n;
    int aEnd = aBegin + n - 1;
    int aStride = TILE_SIZE;
    
    int bBegin = TILE_SIZE * bx;
    int bStride = TILE_SIZE * k;

    int accu = 0;

    // 计算C的一个Tile
    for(int i = aBegin, j = bBegin; i <= aEnd; i += aStride, j += bStride){
        //load share memory
        //从Tile中取出一个点放到共享内存中
        A_tile[ty][tx] = d_A[i + n * ty + tx];
        B_tile[tx][ty] = d_B[j + k * ty + tx];
        //B_tile[tx][ty] = d_B[j + k * tx + ty];
        __syncthreads();

        for(int k = 0;k < TILE_SIZE; k++)
            accu += A_tile[ty][k] * B_tile[k][tx];
        
        __syncthreads();
    }
    //A中横着的一行和B中竖着的一列累加完毕放到C中对应位置
    int cIdx = k * TILE_SIZE * by + TILE_SIZE * bx;
    d_C[cIdx + k * ty + tx] = accu;
}
