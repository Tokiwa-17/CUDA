#include <stdio.h>
#include <cuda_runtime.h>
#include "matrix.cuh"

/*
*********************************************************************
function name : gpuMatrixMul
description : multiplication of two matrix
parameters :
    &d_A GPU device pointer to a (m, n) matrix(A)
    &d_B GPU device pointer to a (n, k) matrix(B)
    &d_C GPU device output pointer to a (m, k) matrix(C)
return: none
*********************************************************************
*/
__global__ void gpuMatrixMul(int* d_A, int* d_B, int* d_C, int m, int n, int k) {
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    int col = threadIdx.y + blockDim.y * blockIdx.y;
    int sum = 0;
    if (row < m && col < k) {
        for (int i = 0;i < n;i++)
            sum += d_A[row * n + i] * d_B[i * k + col];
        d_C[row * k + col] = sum;
    }
}

/*
*********************************************************************
function name: cpuMatrixMul
description: Multiplication two matrix in CPU.
parameters: 
    &h_A CPU host pointer to a (m, n) matrix (A)
    &h_B CPU host pointer to a (n, k) matrix (B)
    &h_C CPU host output pointer to a (m, k) matrix (C) 
    to store the result
return: none
*********************************************************************
*/
void cpuMatrixMul(int *h_A, int * h_B, int* h_C, int m, int n, int k){
    for(int i = 0;i < m;i++)
        for(int j = 0;j < k;j++){
            int sum = 0;
            for(int l = 0;l < n;l++)
                sum += h_A[i * n + l] * h_B[l * k + j];
            h_C[i * k + j] = sum;
        }
}
