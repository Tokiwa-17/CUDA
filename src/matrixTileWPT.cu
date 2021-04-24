#include <stdio.h>
#include <cuda_runtime.h>
#include "../include/config.cuh"
#include "../include/matrixTileWPT.cuh"



//矩阵的大小设置成TILE_SIZE 的倍数
/*__global__ void gpuMatrixMulTileWPT(int* d_A, int* d_B, int* d_C, int m, int n, int k){
    
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

    int accu[WPT];
    for(int i = 0; i < WPT; i++) accu[i] = 0;

    //计算C的一个Tile
    for(int i = aBegin, j = bBegin; i <= aEnd; i += aStride, j += bStride){
        //load share memory
        //从全局地址取出一个点放到共享内存中
        A_tile[ty][tx] = d_A[i + n * ty + tx];
        for(int l = 0; l < WPT; l++)
            B_tile[tx][ty * WPT] = d_B[j + k * tx + ty + l];

        __syncthreads();

        for(int l = 0; l < WPT; l++){
            for(int t = 0;t < TILE_SIZE; t++)
                accu[l] += A_tile[ty][t] * B_tile[t][tx + l];
        }
        
        __syncthreads();
    }
    //A中横着的一行和B中竖着的一列累加完毕放到C中对应位置
    int cIdx = k * TILE_SIZE * by + TILE_SIZE * bx;
    for(int l = 0; l < WPT; l++)
        d_C[cIdx + k * ty + tx + l] = accu[l];
}
*/

__global__ void gpuMatrixMulTileWPT(int* A, int* B, int* C, int m, int n, int k){
    
    // Thread identifiers
    int row = threadIdx.x, col = threadIdx.y;
    int globalRow = TILE_SIZE * blockIdx.x + row;
    int globalCol = TILE_SIZE * blockIdx.y + col;
    int rts = TILE_SIZE / WPT;

    // Local memory to fit a tile of TILE_SIZE * TILE_SIZE of A and B
    __shared__ int A_tile[TILE_SIZE][TILE_SIZE];
    __shared__ int B_tile[TILE_SIZE][TILE_SIZE];

    // initialize arrays
    int acc[WPT];
    for(int i = 0; i < WPT; i++) acc[i] = 0;

    const unsigned int numTiles = k / TILE_SIZE;
    for(int t = 0; t < numTiles; t++){

        for(int w = 0; w < WPT; w++){
            const unsigned int tileRow = TILE_SIZE * t + row;
            const unsigned int tileCol = TILE_SIZE * t + col;
            A_tile[col + w * rts][row] = A[(tileCol + w * rts) * m + globalRow];
            B_tile[col + w * rts][row] = B[(globalRow + w * rts) * k + tileRow];
        }

        __syncthreads();

        for(int j = 0; j < TILE_SIZE; j++){
            for(int w = 0; w < WPT; w++)
                acc[w] += A_tile[j][row] * B_tile[col + w * rts][j];
        }

        __syncthreads();
    }

    for(int w = 0; w < WPT; w++)
        C[(globalCol + w * rts) * m + globalRow] = acc[w];
}
