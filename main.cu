// Include C++ header files.
#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include "./include/config.cuh"
#include "./include/util.cuh"
#include "./include/gpuMatrixCublas.cuh"
#include "./include/matrixNaive.cuh"
#include "./include/matrixTile.cuh"
#include "./include/matrixCoalescing.cuh"
#include "./include/matrixBankConflict.cuh"
#include "./include/matrixTileWPT.cuh"
#include "./include/matrixTranspose.cuh"
#include "./include/matrixComOpt.cuh"
#include "./include/cpuMatrixStrassen.cuh"
#include "./include/matrixPrefetch.cuh"
//#include "./include/gpuMatrixStrassen.cuh"
using namespace std;
// Include local CUDA header files.

int main(int argc, char ** argv){


    // set up device
    int dev = 0;
    initDevice(dev);

    // input m, n, k
    int m = 32, n = 32, k = 32;
    if(argc > 1) m = atoi(argv[1]);
    if(m  % 32) {
        cout << "The input must be a multiple number of 32!\n";
        return 0;
    }
    n = k = m;
    // Allocate memory space on the host
    int *h_A = (int*)malloc(sizeof(int) * (m * n));
    int *h_B = (int*)malloc(sizeof(int) * (n * k));
    int *h_BT =(int*)malloc(sizeof(int) * (n * k));
    int *h_C = (int*)malloc(sizeof(int) * (m * k));
    int *h_odata = (int*)malloc(sizeof(int) * (m * k));

    // Initialize 
    initialDataInt(h_A, m * n);
    initialDataInt(h_B, n * k);
    matrixTranspose(h_B, h_BT, n, k);
 
    // Allocate memory space on the device
    int *d_A, *d_B, *d_BT, *d_C;
    cudaMalloc((void**)&d_A, sizeof(int) * (m * n));
    cudaMalloc((void**)&d_B, sizeof(int) * (n * k));
    cudaMalloc((void**)&d_BT,sizeof(int) * (n * k));
    cudaMalloc((void**)&d_C, sizeof(int) * (m * k));

    // Copy matrix A and B from host to device memory
    cudaMemcpy(d_A, h_A, sizeof(int) * (m * n), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(int) * (n * k), cudaMemcpyHostToDevice);
    cudaMemcpy(d_BT,h_BT,sizeof(int) * (n * k), cudaMemcpyHostToDevice);

    // CPU Matrix multiplication
    double iStart = cpuSecond();
    cpuMatrixMul(h_A, h_B, h_C, m, n, k);
    double iElaps = cpuSecond() - iStart;   
    printf("cpu Matrix multiplication\t\telapsed %f sec.\n", iElaps);

    // CPU Matrix multiplication by Strassen
    Matrix a(h_A, n), b(h_B, n);
    iStart = cpuSecond();
    Matrix c = strassen(a, b);
    iElaps = cpuSecond() - iStart;  
    printf("cpu Matrix multiplication by Strassen\telapsed %f sec.\n", iElaps);
    c.checkResult(h_C); 

    // GPU Matrix Benchmark
    float alpha = 1.0f, beta = 0.0f;
    gpuMatrixCublas(h_A, h_B, h_C, m, n, k, m, n, k, alpha, beta);

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

    printf("gpu Matrix multiplication2\t\telapsed %f sec. <<<grid %d block "
    "%d>>>\n", iElaps, grid.x, block.x);
    checkResult(h_C, h_odata, m * k);

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

    // GPU Matrix multiplication by Coalescing
    block.x = TILE_SIZE, block.y = TILE_SIZE;
    grid.x = k / TILE_SIZE, grid.y = m / TILE_SIZE;
    iStart = cpuSecond();
    gpuMatrixMulCoalescing<<<grid, block>>>(d_A, d_BT, d_C, m, n, k);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    iElaps = cpuSecond() - iStart;
    CHECK(cudaMemcpy(h_odata, d_C, sizeof(int) *(m * k), cudaMemcpyDeviceToHost));

    printf("gpu Matrix multiplication3\t\telapsed %f sec. <<<grid %d block "
    "%d>>>\n", iElaps, grid.x, block.x);
    checkResult(h_C, h_odata, m * k);

    // GPU Matrix multiplication by avoiding share memory bank conflict
    block.x = TILE_SIZE, block.y = TILE_SIZE;
    grid.x = k / TILE_SIZE, grid.y = m / TILE_SIZE;
    iStart = cpuSecond();
    gpuMatrixMulBankConflict<<<grid, block>>>(d_A, d_BT, d_C, m, n, k);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    iElaps = cpuSecond() - iStart;
    CHECK(cudaMemcpy(h_odata, d_C, sizeof(int) *(m * k), cudaMemcpyDeviceToHost));

    printf("gpu Matrix multiplication4\t\telapsed %f sec. <<<grid %d block "
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
    printf("gpu Matrix multiplication5\t\telapsed %f sec. <<<grid %d block "
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
    printf("gpu Matrix multiplication5(WPT = 4)\telapsed %f sec. <<<grid %d block "
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
    printf("gpu Matrix multiplication5(WPT = 8)\telapsed %f sec. <<<grid %d block "
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
        printf("gpu Matrix multiplication6\t\telapsed %f sec. <<<grid %d block "
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
        printf("gpu Matrix multiplication6\t\telapsed %f sec. <<<grid %d block "
        "%d>>>\n", iElaps, grid.x, block.x);
        checkResult(h_C, h_odata, m * k);
    }
    
    // GPU Matrix multiplication by prefetching
    //block.x = TILE_SIZE, block.y = 8;
    //grid.x = k / (TILE_SIZE * 8), grid.y = m / TILE_SIZE;
    if(m > 32){

        block.x = TILE_SIZE, block.y = VEC_SIZE;
        grid.x = k / (TILE_SIZE * VEC_SIZE), grid.y = m / TILE_SIZE;
        iStart = cpuSecond();
        gpuMatrixMulPrefetch<<<grid, block>>>(d_A, d_B, d_C, m, n, k);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());
        iElaps = cpuSecond() - iStart;
        CHECK(cudaMemcpy(h_odata, d_C, sizeof(int) *(m * k), cudaMemcpyDeviceToHost));

        printf("gpu Matrix multiplication7\t\telapsed %f sec. <<<grid %d block "
        "%d>>>\n", iElaps, grid.x, block.x);
        checkResult(h_C, h_odata, m * k);
    }


    free(h_A);
    free(h_B);
    free(h_BT);
    free(h_C);
    free(h_odata);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_BT);
    cudaFree(d_C);

    return 0;
}
