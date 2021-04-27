#include "../include/config.cuh"
#include "../include/til.cuh"
#include "../include/matrixTranspose.cuh"

void matrixTranspose(int *A, int *B, int m, int n){

    // 输入矩阵A, 输出矩阵B = A ^ T.
    cublasHandle_t handle;
    cublasCreate(&handle);

    printMatrix(A, m, n);
    printMatrix(B, m, n);

    int *d_A, *d_B;
    CHECK(cudaMalloc((void **)&d_A, sizeof(int) * (m * n)));
    CHECK(cudaMalloc((void **)&d_B, sizeof(int) * (m * n)));
    
    CHECK(cudaMemcpy(d_A, A, sizeof(int) * (m * n), cudaMemcpyHostToDevice));

    float *f_A, *f_B;
    CHECK(cudaMalloc((void **)&f_A, sizeof(float) * (m * n)));
    CHECK(cudaMalloc((void **)&f_B, sizeof(float) * (m * n)));

    dim3 block(BLOCK_SIZE, BLOCK_SIZE), grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);

    intPtrToFloatPtr<<<grid, block>>>(d_A, f_A, m, n);

    float alpha = 1.f, beta = 0.f;

    cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, &alpha, f_A, m, &beta, f_B, m, f_B, m);

    /*********DEBUG********
    float* testA, *testB;
    testA = (float*)malloc(sizeof(float) * (m * n));
    testB = (float*)malloc(sizeof(float) * (m * n));
    CHECK(cudaMemcpy(testA, f_A, sizeof(float) * (m * n), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(testB, f_B, sizeof(float) * (m * n), cudaMemcpyDeviceToHost));
    printMatrix(testA, m, n);
    printMatrix(testB, m, n);
    ***********************/

    floatPtrToIntPtr<<<grid, block>>>(f_B, d_B, m, n);

    CHECK(cudaMemcpy(B, d_B, sizeof(int) * (m * n), cudaMemcpyDeviceToHost));

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(f_A);
    cudaFree(f_B);
}
