#include <stdio.h>
#include <cuda_runtime.h>
#include "../include/til.cuh"
#include "../include/config.cuh"

void cpuMatrixMul(int *h_A, int * h_B, int* h_C, int m, int n, int k){
    for(int i = 0;i < m;i++)
        for(int j = 0;j < k;j++){
            int sum = 0;
            for(int l = 0;l < n;l++)
                sum += h_A[i * n + l] * h_B[l * k + j];
            h_C[i * k + j] = sum;
        }
}

#ifdef _WIN32
int gettimeofday(struct timeval* tp, void* tzp){
    time_t clock;
    struct tm tm;
    SYSTEMTIME wtm;
    GetLocalTime(&wtm);
    tm.tm_year = wtm.wYear - 1900;
    tm.tm_mon = wtm.wMonth - 1;
    tm.tm_mday = wtm.wDay;
    tm.tm_hour = wtm.wHour;
    tm.tm_min = wtm.wMinute;
    tm.tm_sec = wtm.wSecond;
    tm.tm_isdst = -1;
    clock = mktime(&tm);
    tp->tv_sec = clock;
    tp->tv_usec = wtm.wMilliseconds * 1000;
    return (0);
}
#endif

double cpuSecond(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}

void initialData(float* ip, int size){
    //generate different seed for random number
    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0;i < size;i++){
        ip[i] = (float)(rand() & 0xffff) / 1000.0f;
    }
}

void initialDataInt(int* ip, int size){
    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < size; i++){
        ip[i] = int((rand() % 100) - 50);
    }
}

void printMatrix(int* C, const int nx, const int ny){
    int* ic = C;
    printf("Matrix<%d,%d>:\n", ny, nx);
    for (int i = 0;i < ny;i++){
        for (int j = 0;j < nx;j++){
            printf("%6d ", C[j]);
        }
        ic += nx;
        printf("\n");
    }
}


void initDevice(int devNum){
    int dev = devNum;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

}

void checkResult(int* hostRef, int* gpuRef, const int N){
    double epsilon = 1.0E-8;
    for (int i = 0;i < N;i++){
        if (abs(hostRef[i] - gpuRef[i]) > epsilon){
            printf("Results do not match!\n");
            printf("%d(hostRef[%d] )!= %d(gpuRef[%d])\n", hostRef[i], i, gpuRef[i], i);
            return;
        }
    }
    printf("Check result success!\n");
}
