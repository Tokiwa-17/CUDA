#pragma once
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cmath>
#include <cassert>

void gpuMatrixCublas(int* d_A, int* d_B, int* d_C, int m, int n, int k);
