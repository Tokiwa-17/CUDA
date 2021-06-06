#pragma once
#include <time.h>
#ifdef _WIN32
#	include <windows.h>
#else
#	include <sys/time.h>
#endif
#define CHECK(call){\
  const cudaError_t error=call;\
  if(error!=cudaSuccess){\
      printf("ERROR: %s:%d,",__FILE__,__LINE__);\
      printf("code:%d,reason:%s\n",error,cudaGetErrorString(error));\
      exit(1);\
  }\
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
void cpuMatrixMul(int *h_A, int * h_B, int* h_C, int m, int n, int k);


int gettimeofday(struct timeval* tp, void* tzp);
//#endif

double cpuSecond();

void initialData(float* ip, int size);

void initialDataInt(int* ip, int size);

void printMatrix(int* C, const int nx, const int ny);

void printMatrix(float* C, const int nx, const int ny);

void initDevice(int devNum);

void checkResult(int* hostRef, int* gpuRef, const int N);

__global__ void intPtrToFloatPtr(int *in, float *out, unsigned int m, unsigned int n);

__global__ void floatPtrToIntPtr(float *in, int *out, unsigned int m, unsigned int n);


