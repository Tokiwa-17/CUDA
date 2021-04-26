#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include "../include/gpuMatrixCublas.cuh"
#include "../include/config.cuh"
#include "../include/til.cuh"
using namespace std;

__global__ void intPtrToFloatPtr(int *in, float* out, unsigned int m, unsigned int n){
    unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;

    out[idx] = in[idx] * 1.0f;
}

__global__ void floatPtrToIntPtr(float *in, int* out, unsigned int m, unsigned int n){
    unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;

    out[idx] = (int)in[idx];
}

void gpuMatrixCublas(int* A, int* B, int* C, int lda, int ldb, int ldc,
                     int m, int n, int k, float alpha, float beta){
    
    //cudaStatus_t stat;
    cublasHandle_t handle;
    cublasCreate(&handle);

    int *d_A, *d_B;
    CHECK(cudaMalloc((void**)&d_A, sizeof(int) * (m * n)));
    CHECK(cudaMalloc((void**)&d_B, sizeof(int) * (n * k)));
    CHECK(cudaMemcpy(d_A, A, sizeof(int) * (m * n), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, B, sizeof(int) * (n * k), cudaMemcpyHostToDevice));

    float* f_A, *f_B, *f_C, *f_odata;
    CHECK(cudaMalloc((void**)&f_A, sizeof(float) * (m * n)));
    CHECK(cudaMalloc((void**)&f_B, sizeof(float) * (n * k)));
    CHECK(cudaMalloc((void**)&f_C, sizeof(float) * (m * k)));
    f_odata = (float*)malloc(sizeof(float) * (m * k));

    int *f_odataCopy;
    f_odataCopy = (int*)malloc(sizeof(int) * (m * k));

    dim3 block(m, 1), grid(n, 1);

    intPtrToFloatPtr<<<grid, block>>>(d_A, f_A, m, n);
    intPtrToFloatPtr<<<grid, block>>>(d_B, f_B, n, k);
    cudaDeviceSynchronize();

    /********DEBUG********
    float* test;
    test = (float*)malloc(sizeof(float) * (m * n));
    CHECK(cudaMemcpy(test, f_A, sizeof(float) * (m * n), cudaMemcpyDeviceToHost));
    printMatrix(A, m, n);
    printMatrix(test, m, n);
    *********************/

    double iStart = cpuSecond();
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, k, n, 
        &alpha, f_B, ldb, f_A, lda, &beta, f_C, ldc);
    //cublas 列主序所以调换f_B和f_A的位置
    double iElaps = cpuSecond() - iStart;
    printf("gpu Matrix Benchmark(Cublas)\t\telapsed %f sec.\n", iElaps);

    cublasGetMatrix(m, k, sizeof(float), f_C, m, f_odata, m);
    //floatPtrToIntPtr<<<grid, block>>>(f_odata, f_odataCopy, m, k);
    printMatrix(f_odata, m, k);
    printMatrix(C, m, k);
    checkResult(C, f_odataCopy, m);
    
    
    cudaFree(f_A);
    cudaFree(f_B);
    cudaFree(f_C);
    free(f_odata);
    free(f_odataCopy);
}
