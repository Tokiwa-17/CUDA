#include <stdio.h>
#include <cuda_runtime.h>
#include "../include/config.cuh"
#include "../include/matrixComOpt.cuh"

// ATile: TILE_SIZE * TILE_SIZE
// BTile: TILE_SIZE * (TILE_SIZE * VEC_SIZE)


//每个线程计算C的竖着的一条向量，所以我们需要加载一个B值和一列长度为VEC_SIZE的向量C到寄存器中
__global__ void gpuMatrixComOpt(int *A, int *B, int *C, int m, int n, int k){
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    __shared__ int ATile[TILE_SIZE * TILE_SIZE];
    volatile int cCol[TILE_SIZE];
    for(int i = 0; i < TILE_SIZE; i++) cCol[i] = 0;

    int aBegin = n * TILE_SIZE * by;
    int aEnd = aBegin + n - 1;
    int aStride = TILE_SIZE;

    int bBegin = TILE_SIZE * VEC_SIZE * bx;
    int bStride = TILE_SIZE * k;

    for(int a = aBegin, b = bBegin; a <= aEnd; a += aStride, b += bStride){
        //计算C的一个点需要A的行向量和B的列向量，所以遍历方式不变
        //Step 1:load A_{(0, 0)} to shared memory.
        //
        // 因为共享内存可以在线程之间共享，我们每个线程块有<TILE_SIZE, VEC_SIZE>个线程，
        // 所以每个线程加载TILE_SIZE / VEC_SIZE 个值
        //
        for(int i = 0; i < TILE_SIZE / VEC_SIZE; i++)
            ATile[(i * VEC_SIZE + ty) + TILE_SIZE * tx] = A[a + n * (i * VEC_SIZE + ty) + tx];
        //实际上i == 0时把A的四个列相距VEC_SIZE的值放到ATile的行相距VEC_SIZE的位置上
        

        __syncthreads();

        int *aPtr = ATile;
        int *bPtr = &B[b + TILE_SIZE * ty + tx];
        //从B的全局坐标中取出值放入寄存器中

        for(int i = 0; i < TILE_SIZE; i++){
            int bVal = * bPtr;
            for(int j = 0; j < TILE_SIZE; j++)
                cCol[j] += aPtr[j] * bVal;
                //因为ATile相当于转置过，所以直接对应相乘即可
            aPtr += TILE_SIZE;
            bPtr += k;
            //把原来一次性的结果分散到多次计算
        }
        __syncthreads();
    }
    int cPos = k * TILE_SIZE * by + TILE_SIZE * VEC_SIZE * bx + TILE_SIZE * ty + tx;
    //每个线程块计算<TILE_SIZE, TILE_SIZE * VEC_SIZE>大小的C.
    //每个线程大小<TILE_SIZE, VEC_SIZE>, 所以每个线程计算VEC_SIZE个数值
    //
    for(int i = 0;i < TILE_SIZE; i++){
        C[cPos] = cCol[i];
        cPos += k;
    }
}
