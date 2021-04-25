// Include C++ header files.
#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include "./include/config.cuh"
#include "./include/til.cuh"
#include "./include/matrixNaive.cuh"
#include "./include/matrixTile.cuh"
#include "./include/matrixTileWPT.cuh"
#include "./include/matrixTranspose.cuh"
#include "./include/matrixComOpt.cuh"
#include "./include/cpuMatrixStrassen.cuh"
//#include "./include/cublas.cuh"

// Include local CUDA header files.

int main(int argc, char ** argv){
    // set up device
    int dev = 0;
    initDevice(dev);

    // input m, n, k
    int m = 320, n = 320, k = 320;
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

    // Allocate memory space on the device
    int *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, sizeof(int) * (m * n));
    cudaMalloc((void**)&d_B, sizeof(int) * (n * k));
    cudaMalloc((void**)&d_C, sizeof(int) * (m * k));

    // Copy matrix A and B from host to device memory
    cudaMemcpy(d_A, h_A, sizeof(int) * (m * n), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(int) * (n * k), cudaMemcpyHostToDevice);

    // CPU Matrix multiplication
    double iStart = cpuSecond();
    cpuMatrixMul(h_A, h_B, h_C, m, n, k);
    double iElaps = cpuSecond() - iStart;   
    printf("cpu Matrix multiplication\t\telapsed %f sec.\n", iElaps);

    // CPU Matrix multiplication by Strassen
    /*cpuMatrixStrassen Strassen(h_A, h_B, m);
    double iStart = cpuSecond();
    ;
    cpuMatrixMul(h_A, h_B, h_C, m, n, k);
    double iElaps = cpuSecond() - iStart;   
    printf("cpu Matrix multiplication\t\telapsed %f sec.\n", iElaps);*/

    // GPU Matrix multiplication
    unsigned int gridRows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int gridCols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 grid(gridRows, gridCols);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);

    iStart = cpuSecond();
    gpuMatrixMul<< <grid, block >> > (d_A, d_B, d_C, m, n, k);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    iElaps = cpuSecond() - iStart;
    CHECK(cudaMemcpy(h_odata, d_C, sizeof(int) * (m * k), cudaMemcpyDeviceToHost));

    printf("gpu Matrix multiplication\t\telapsed %f sec. <<<grid %d block "
        "%d>>>\n", iElaps, grid.x, block.x);

    // Check result
    checkResult(h_C, h_odata, m * k);

    //cublas(d_A, d_B, d_C, m, n, k);

    // GPU Matrix multiplication by tile
    block.x = TILE_SIZE, block.y = TILE_SIZE;
    grid.x = k / TILE_SIZE, grid.y = m / TILE_SIZE;
    if(grid.x == 0 || grid.y == 0){
        unsigned int gridRows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
        unsigned int gridCols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
        dim3 grid(gridRows, gridCols);
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);

        iStart = cpuSecond();
        gpuMatrixMul<< <grid, block >> > (d_A, d_B, d_C, m, n, k);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());
        iElaps = cpuSecond() - iStart;
    }
    else{
        iStart = cpuSecond();
        gpuMatrixMulTile<<<grid, block>>>(d_A, d_B, d_C, m, n, k);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());
        iElaps = cpuSecond() - iStart;
        CHECK(cudaMemcpy(h_odata, d_C, sizeof(int) *(m * k), cudaMemcpyDeviceToHost));
    }
    //printMatrix(h_odata, m, k);

    printf("gpu Matrix multiplication2\t\telapsed %f sec. <<<grid %d block "
    "%d>>>\n", iElaps, grid.x, block.x);
    checkResult(h_C, h_odata, m * k);

    // GPU Matrix multiplication by tile, optimized by WPT
    block.x = TILE_SIZE / WPT, block.y = TILE_SIZE;
    grid.x = k / TILE_SIZE, grid.y = m / TILE_SIZE;
    iStart = cpuSecond();
    gpuMatrixMulTileWPT<<<grid, block>>>(d_A, d_B, d_C, m, n, k);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    iElaps = cpuSecond() - iStart;
    CHECK(cudaMemcpy(h_odata, d_C, sizeof(int) *(m * k), cudaMemcpyDeviceToHost));
    printf("gpu Matrix multiplication3\t\telapsed %f sec. <<<grid %d block "
    "%d>>>\n", iElaps, grid.x, block.x);
    checkResult(h_C, h_odata, m * k);

    // GPU Matrix multiplication by tile, optimized by WPT = 4
    block.x = TILE_SIZE / 4, block.y = TILE_SIZE;
    grid.x = k / TILE_SIZE, grid.y = m / TILE_SIZE;
    iStart = cpuSecond();
    gpuMatrixMulTileWPTop4<<<grid, block>>>(d_A, d_B, d_C, m, n, k);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    iElaps = cpuSecond() - iStart;
    CHECK(cudaMemcpy(h_odata, d_C, sizeof(int) *(m * k), cudaMemcpyDeviceToHost));
    printf("gpu Matrix multiplication3(WPT = 4)\telapsed %f sec. <<<grid %d block "
    "%d>>>\n", iElaps, grid.x, block.x);
    checkResult(h_C, h_odata, m * k);

    // GPU Matrix multiplication by tile, optimized by WPT = 8
    block.x = TILE_SIZE / 8, block.y = TILE_SIZE;
    grid.x = k / TILE_SIZE, grid.y = m / TILE_SIZE;
    iStart = cpuSecond();
    gpuMatrixMulTileWPTop8<<<grid, block>>>(d_A, d_B, d_C, m, n, k);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    iElaps = cpuSecond() - iStart;
    CHECK(cudaMemcpy(h_odata, d_C, sizeof(int) *(m * k), cudaMemcpyDeviceToHost));
    printf("gpu Matrix multiplication3(WPT = 8)\telapsed %f sec. <<<grid %d block "
    "%d>>>\n", iElaps, grid.x, block.x);
    checkResult(h_C, h_odata, m * k);

    // GPU Matrix multiplication by tile, optimized by Computational optimization4
    if(m > 32){
        block.x = TILE_SIZE, block.y = VEC_SIZE;
        grid.x = k / (TILE_SIZE * VEC_SIZE), grid.y = m / TILE_SIZE;
        //grid.x = (k + TILE_SIZE - 1) / TILE_SIZE, grid.y = (m + TILE_SIZE * VEC_SIZE - 1) / (TILE_SIZE * VEC_SIZE);
        iStart = cpuSecond();
        gpuMatrixComOpt<<<grid, block>>>(d_A, d_B, d_C, m, n, k);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());
        iElaps = cpuSecond() - iStart;
        CHECK(cudaMemcpy(h_odata, d_C, sizeof(int) *(m * k), cudaMemcpyDeviceToHost));
        printf("gpu Matrix multiplication4\t\telapsed %f sec. <<<grid %d block "
        "%d>>>\n", iElaps, grid.x, block.x);
        checkResult(h_C, h_odata, m * k);
    }

    // GPU Matrix multiplication by tile, optimized by Computational optimization8
    if(m > 64){
        block.x = TILE_SIZE, block.y = 8;
        grid.x = k / (TILE_SIZE * 8), grid.y = m / TILE_SIZE;
        iStart = cpuSecond();
        gpuMatrixComOpt8<<<grid, block>>>(d_A, d_B, d_C, m, n, k);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());
        iElaps = cpuSecond() - iStart;
        CHECK(cudaMemcpy(h_odata, d_C, sizeof(int) *(m * k), cudaMemcpyDeviceToHost));
        printf("gpu Matrix multiplication4\t\telapsed %f sec. <<<grid %d block "
        "%d>>>\n", iElaps, grid.x, block.x);
        checkResult(h_C, h_odata, m * k);
    }
    return 0;
}
