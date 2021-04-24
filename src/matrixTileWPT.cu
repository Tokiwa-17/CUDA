#include <stdio.h>
#include <cuda_runtime.h>
#include "../include/config.cuh"
#include "../include/matrixTileWPT.cuh"



//矩阵的大小设置成TILE_SIZE 的倍数
__global__ void gpuMatrixMulTileWPT(int* d_A, int* d_B, int* d_C, int m, int n, int k){
    
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

    volatile int accu[WPT];
    for(int i = 0; i < WPT; i++) accu[i] = 0;

    //计算C的一个Tile
    for(int i = aBegin, j = bBegin; i <= aEnd; i += aStride, j += bStride){
        //load share memory
        //从全局地址取出一个点放到共享内存中
        A_tile[ty][2 * tx] = d_A[i + n * ty + tx * 2];
        A_tile[ty][2 * tx + 1] = d_A[i + n * ty + tx * 2 + 1];
        B_tile[2 * tx][ty] = d_B[j + k * (2 * tx) + ty];
        B_tile[2 * tx + 1][ty] = d_B[j + k * (2 * tx + 1) + ty];

        __syncthreads();

            for(int t = 0;t < TILE_SIZE; t++){
                accu[0] += A_tile[ty][t] * B_tile[t][2 * tx];
                accu[1] += A_tile[ty][t] * B_tile[t][2 * tx + 1];
            }
        
        __syncthreads();
    }
    //A中横着的一行和B中竖着的一列累加完毕放到C中对应位置
    int cIdx = k * TILE_SIZE * by + TILE_SIZE * bx;

    d_C[cIdx + k * ty + 2 * tx] = accu[0];
    d_C[cIdx + k * ty + 2 * tx + 1] = accu[1];

}

//矩阵的大小设置成TILE_SIZE 的倍数
__global__ void gpuMatrixMulTileWPTop4(int* d_A, int* d_B, int* d_C, int m, int n, int k){
    
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

    volatile int accu[4];
    for(int i = 0; i < 4; i++) accu[i] = 0;

    //计算C的一个Tile
    for(int i = aBegin, j = bBegin; i <= aEnd; i += aStride, j += bStride){
        //load share memory
        //从全局地址取出一个点放到共享内存中
        A_tile[ty][4 * tx] = d_A[i + n * ty + tx * 4];
        A_tile[ty][4 * tx + 1] = d_A[i + n * ty + tx * 4 + 1];
        A_tile[ty][4 * tx + 2] = d_A[i + n * ty + tx * 4 + 2];
        A_tile[ty][4 * tx + 3] = d_A[i + n * ty + tx * 4 + 3]
        B_tile[4 * tx][ty] = d_B[j + k * (4 * tx) + ty];
        B_tile[4 * tx + 1][ty] = d_B[j + k * (4 * tx + 1) + ty];
        B_tile[4 * tx + 2][ty] = d_B[j + k * (4 * tx + 2) + ty];
        B_tile[4 * tx + 3][ty] = d_B[j + k * (4 * tx + 3) + ty];

        __syncthreads();

            for(int t = 0;t < TILE_SIZE; t++){
                accu[0] += A_tile[ty][t] * B_tile[t][4 * tx];
                accu[1] += A_tile[ty][t] * B_tile[t][4 * tx + 1];
                accu[2] += A_tile[ty][t] * B_tile[t][4 * tx + 2];
                accu[3] += A_tile[ty][t] * B_tile[t][4 * tx + 3];
            }
        
        __syncthreads();
    }
    //A中横着的一行和B中竖着的一列累加完毕放到C中对应位置
    int cIdx = k * TILE_SIZE * by + TILE_SIZE * bx;

    d_C[cIdx + k * ty + 4 * tx] = accu[0];
    d_C[cIdx + k * ty + 4 * tx + 1] = accu[1];
    d_C[cIdx + k * ty + 4 * tx + 2] = accu[2];
    d_C[cIdx + k * ty + 4 * tx + 3] = accu[3];
}
