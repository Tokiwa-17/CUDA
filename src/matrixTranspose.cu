#include <stdio.h>
#include <cuda_runtime.h>
#include "../include/config.cuh"
#include "../include/matrixTranspose.cuh"

__global__ void matrixNaiveTrans(int *out, int *in, const int nx, const int ny){
    // coordinate(ix, iy)
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if(ix < nx && iy < ny)
        out[ix * ny + iy] = in[iy * nx + ix];
}

__global__ void matrixTranspose(int *out, int *in, int nx, int ny){
    __shared__ int tile[BDIMY][BDIMX];
    unsigned int ix, iy, ti, to;
    //计算线程的全局坐标
    ix = blockIdx.x * blockDim.x + threadIdx.x;
    iy = blockIdx.y * blockDim.y + threadIdx.y;
    ti = iy * nx + ix;

    // bidx表示线程在这个线程块中的位置，计算新的坐标位置
    unsigned int bidx, irow, icol;
    bidx = threadIdx.y * blockDim.x + threadIdx.x;
    irow = bidx / blockDim.y;
    icol = bidx % blockDim.y;

    // 计算转置后的全局坐标
    ix = blockIdx.y * blockDim.y + icol;
    iy = blockIdx.x * blockDim.x + irow;
    to = iy * ny + ix;

    if(ix < nx && iy < ny){
        tile[threadIdx.y][threadIdx.x] = in[ti];
        __syncthreads();
        out[to] = tile[icol][irow];
    }

}
