#pragma once
#include <stdio.h>
#include <cuda_runtime.h>
#define BDIMY 32
#define BDIMX 16

/*
*********************************************************************
function name : matrixNaiveTrans
description : a naive complementation of transpose
parameters :
int *out, int *in, int nx, int ny
return: none
*********************************************************************
*/
__global__ void matrixNaiveTrans(int *out, int *in, const int nx, const int ny);
/*
*********************************************************************
function name : matrixTranspose
description : derive transpose of a matrix by share memory
parameters :

steps:
1. 从全局内存中读取块的一行，并向共享内存中写入一行
2. 从共享内存中读取一列，并向全局内存中写入块中的一行
return: none
*********************************************************************
*/
__global__ void matrixTranspose(int *out, int *in, int nx, int ny);
