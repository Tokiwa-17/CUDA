#include "../include/config.cuh"
#include "../include/til.cuh"
#include "../include/matrixTranspose.cuh"

__global__ void intPtrToFloatPtr(int *in, float* out, unsigned int m, unsigned int n){
    unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;

    out[idx] = in[idx] * 1.0f;
}

__global__ void floatPtrToIntPtr(float *in, int* out, unsigned int m, unsigned int n){
    unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;

    out[idx] = (int)in[idx];
}

void matrixTranspose(int *A, int *B, int m, int n){

    // 输入矩阵A, 输出矩阵B = A ^ T.
    cublasHandle_t handle;
    cublasCreate(&handle);

    int *d_A, *d_B;
    CHECK(cudaMalloc((void **)&d_A, sizeof(int) * (m * n)));
    CHECK(cudaMalloc((void **)&d_B, sizeof(int) * (m * n)));
    
    CHECK(cudaMemcpy(d_A, A, sizeof(int) * (m * n), cudaMemcpyHostToDevice));

    float *f_A, *f_B;
    CHECK(cudaMalloc((void **)&f_A, sizeof(float) * (m * n)));
    CHECK(cudaMalloc((void **)&f_B, sizeof(float) * (m * n)));

    dim3 block(m, 1), grid(n, 1);

    intPtrToFloatPtr<<<grid, block>>>(d_A, f_A, m, n);

    float alpha = 0.f, beta = 1.f;

    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, &alpha, f_A, m, &beta, f_B, m);

    floatPtrToIntPtr<<<grid, block>>>(f_B, d_B, m, n);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(f_A);
    cudaFree(f_B);
}
