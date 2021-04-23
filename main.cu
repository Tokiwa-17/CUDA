// Include C++ header files.
#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include "include/freshman.cuh"
#include "include/matrix.cuh"
// Include local CUDA header files.





int main(int argc, char ** argv){
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s starting reduction at ", argv[0]);
    printf("device %d: %s \n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // input m, n, k
    int m = 10000, n = 10000, k = 10000;
    if(argc > 1) m = atoi(argv[1]);
    if(argc > 2) n = atoi(argv[2]);
    if(argc > 3) k = atoi(argv[3]);

    // Allocate memory space on the host
    int *h_A = (int*)malloc(sizeof(int) * (m * n));
    int *h_B = (int*)malloc(sizeof(int) * (n * k));
    int *h_C = (int*)malloc(sizeof(int) * (m * k));
    int *h_odata = (int*)malloc(sizeof(int) * (m * k));

    // Initialize 
    initialDataInt(h_A, m * n);
    initialDataInt(h_B, n * k);
    initialDataInt(h_C, m * k);

    // Allocate memory space on the device
    int *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, sizeof(int) * (m * n));
    cudaMalloc((void**)&d_B, sizeof(int) * (n * k));
    cudaMalloc((void**)&d_C, sizeof(int) * (m * k));

    // Copy matrix A and B from host to device memory
    cudaMemcpy(d_A, h_A, sizeof(int) * (m * n), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(int) * (n * k), cudaMemcpyHostToDevice);

    unsigned int gridRows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int gridCols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 grid(gridRows, gridCols);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);

    // CPU Matrix multiplication
    double iStart = cpuSecond();
    cpuMatrixMul(h_A, h_B, h_C, m, n, k);
    double iElaps = cpuSecond() - iStart;   
    printf("cpu Matrix multiplication\t\telapsed %f sec.\n", iElaps);

    // GPU Matrix multiplication
    iStart = cpuSecond();
    gpuMatrixMul << <grid, block >> > (d_A, d_B, d_C, m, n, k);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    iElaps = cpuSecond() - iStart;
    CHECK(cudaMemcpy(h_odata, d_C, sizeof(int) * (m * k), cudaMemcpyDeviceToHost));

    printf("gpu Matrix multiplication\t\telapsed %f sec. <<<grid %d block "
        "%d>>>\n", iElaps, grid.x, block.x);

    // Check result
    checkResult(h_C, h_odata, m * k);
    return 0;
}
