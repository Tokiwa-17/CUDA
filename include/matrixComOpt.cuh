#pragma once
#include <stdio.h>
#include <cuda_runtime.h>

// Unlike Tiling, matrix B isn't need to be loaded into shared memory.
// We calculate the outer product of Asub and Bsub, where the size of
// Bsub is define by TILE_SIZE and VECTOR_SIZE. 
// Specifically: 
//   Asub: TILE_SIZE * TILE_SIZE 
//   Bsub: TILE_SIZE * (TILE_SIZE*VECTOR_SIZE)

__global__ void gpuMatrixComOpt(int *A, int *B, int *C, int m, int n, int k);
