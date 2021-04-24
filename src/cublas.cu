#include <stdio.h>
#include <cuda_runtime.h>
#include "../include/cublas.cuh"
#include "../include/config.cuh"
#include "../include/til.cuh"

__global__ void cublas(int* d_A, int* d_B, int* d_C, int m, int n, int k){
    cublasHandle_t handle;
    cublasCreate(&handle);
    float al=1.0f, bet=0;

    double iStart = cpuSecond();
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, 
        &al, dev_a, M, dev_b, K, &bet, dev_c, M);
    double iElaps = cpuSecond() - iStart;

    printf("gpu Matrix Benchmark\t\telapsed %f sec.\n", iElaps);
}
