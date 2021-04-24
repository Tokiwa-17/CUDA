#pragma once
#include <stdio.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

__global__ void cublas(int* d_A, int* d_B, int* d_C, int m, int n, int k);
