#include <stdio.h>
#include <cuda_runtime.h>
#include "../include/gpuMatrixCublas.cuh"
#include "../include/config.cuh"
#include "../include/til.cuh"

void gpuMatrixCublas(float* d_A, float* d_B, float* d_C, int lda, int ldb, int ldc,
                     int m, int n, int k, double alpha, double beta){
    
    cudaStatus_t stat;
    cublasHandle_t handle;
    stat = cublasCreate(&handle);
    
    if(stat != CUBLAS_STATUS_SUCCESS){
        printf("CUBLAS initialization error!\n");
        return;
    }
    double iStart = cpuSecond();
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, k, n, 
        &alpha, B, ldb, A, lda, &beta, C, ldc);
    double iElaps = cpuSecond() - iStart;
    printf("gpu Matrix Benchmark\t\telapsed %f sec.\n", iElaps);
}
