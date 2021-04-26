#include <stdio.h>
#include <cuda_runtime.h>
#include "../include/gpuMatrixCublas.cuh"
#include "../include/config.cuh"
#include "../include/til.cuh"

__global__ void intPtrToFloatPtr(int *in, float* out, unsigned int m, unsigned int n){
    unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;

    out[idx] = in[idx] * 1.0f;
}

void gpuMatrixCublas(int* A, int* B, int* C, int lda, int ldb, int ldc,
                     int m, int n, int k, float alpha, float beta){
    
    //cudaStatus_t stat;
    cublasHandle_t handle;
    cublasCreate(&handle);

    float* f_A, *f_B, *f_C;
    cudaMalloc((void**)&f_A, sizeof(int) * (m * n));
    cudaMalloc((void**)&f_B, sizeof(int) * (n * k));
    cudaMalloc((void**)&f_C, sizeof(int) * (m * k));

    dim3 block(m, 1), grid(n, 1);
    intPtrToFloatPtr<<<grid, block>>>(A, f_A, m, n);
    intPtrToFloatPtr<<<grid, block>>>(B, f_B, n, k);
    cudaDeviceSynchronize();

    double iStart = cpuSecond();
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, k, n, 
        &alpha, f_B, ldb, f_A, lda, &beta, f_C, ldc);
    double iElaps = cpuSecond() - iStart;
    printf("gpu Matrix Benchmark(Cublas)\telapsed %f sec.\n", iElaps);
}
