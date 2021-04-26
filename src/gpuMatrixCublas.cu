#include <stdio.h>
#include <cuda_runtime.h>
#include "../include/gpuMatrixCublas.cuh"
#include "../include/config.cuh"
#include "../include/til.cuh"

void gpuMatrixCublas(float* A, float* B, float* C, int lda, int ldb, int ldc,
                     int m, int n, int k, float alpha, float beta){
    
    //cudaStatus_t stat;
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    double iStart = cpuSecond();
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, k, n, 
        &alpha, B, ldb, A, lda, &beta, C, ldc);
    double iElaps = cpuSecond() - iStart;
    printf("gpu Matrix Benchmark(Cublas)\telapsed %f sec.\n", iElaps);
}
