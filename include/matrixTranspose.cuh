#pragma once
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cmath>
#include <cassert>

void matrixTranspose(int *A, int *B, int m, int n);
