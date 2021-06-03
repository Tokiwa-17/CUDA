#include <stdio.h>
#include <cuda_runtime.h>
#include "../include/config.cuh"
#include "../include/matrixPrefetch.cuh"

/*__global__ void gpuMatrixMulPrefetch(int* d_A, int* d_B, int* d_C, int m, int n, int k){
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    __shared__ int A_tile[TILE_SIZE * TILE_SIZE];
    __shared__ int A_tileNxt[TILE_SIZE * TILE_SIZE];
    
    //register for result of C at each thread
    volatile int cval[TILE_SIZE];
    for(int i = 0;i < TILE_SIZE;i++) cval[i] = 0;

    int aBegin = n * TILE_SIZE * by;
    int aEnd = aBegin + n - 1;
    int aStride = TILE_SIZE;
    
    int bBegin = TILE_SIZE * VEC_SIZE * bx;
    int bStride = TILE_SIZE * k;

    int *cur = A_tile;
    int *nxt = A_tileNxt;
    for(int i = 0;i < TILE_SIZE / VEC_SIZE;i++)
        cur[(i * VEC_SIZE + ty) + TILE_SIZE * tx] = d_A[aBegin + n * (i * VEC_SIZE + ty) + tx];
    
    __syncthreads();

    for(int a = aBegin, b = bBegin; a <= aEnd; a += aStride, b += bStride){
        if(a + aStride <= aEnd){
            for(int i = 0;i < TILE_SIZE / VEC_SIZE; i++)
                nxt[(i * VEC_SIZE) + ty + TILE_SIZE * tx] = d_A[a + n * (i * VEC_SIZE + ty) + tx + aStride];
        }
        int *aptr = cur;
        int *bptr = &d_B[b + TILE_SIZE * ty + tx];

        for(int i = 0;i < TILE_SIZE;i++){
            int bval = *bptr;
            for(int j = 0;j < TILE_SIZE;j++)
                cval[j] += aptr[j] * bval;
            aptr += TILE_SIZE;
            bptr += k;
        }
        __syncthreads();

        int *tmp = cur;
        cur = nxt;
        nxt = tmp;
    }
    int cPos = k * TILE_SIZE * by + TILE_SIZE * VEC_SIZE * bx + TILE_SIZE * ty + tx;
    for(int i = 0;i < TILE_SIZE;i++){
        d_C[cPos] = cval[i];
        cPos += k;
    }
}
*/

__global__ void gpuMatrixMulPrefetch(int *A, int *B, int *C, int M, int K, int N) {
	/* Prefetching method.
	 * Perform outer product of Asub and Bsub.
	 * Specifically:
	 *   Asub: TILE_SIZE * TILE_SIZE
	 *   Bsub: TILE_SIZE * (TILE_SIZE * VEC_SIZE)
	 * 
	 * Before calculating the submatrix, load the next TILE * TILE
	 * submatrix of A into register.
	 *
	 * After calculating, just swap the pointer to exchange the submatrix.
	 */
	int bx = blockIdx.x, by = blockIdx.y;
	int tx = threadIdx.x, ty = threadIdx.y;

	// Allocate As and next_As as column-major array
	__shared__ int As[TILE_SIZE * TILE_SIZE];
	__shared__ int next_As[TILE_SIZE * TILE_SIZE];

	// Allocate register files for sub-result of C at each thread.
	int cv[TILE_SIZE] = {0};

	// Iteration parameters is similar with 
	// computational optimization method.
	int aBegin = K * TILE_SIZE * by;
	int aEnd = aBegin + K - 1;
	int aStep = TILE_SIZE;

	int bBegin = TILE_SIZE * VEC_SIZE * bx;
	int bStep = TILE_SIZE * N;

	int t = VEC_SIZE;
	int *cur = As;
	int *nxt = next_As;
	for (int i = 0; i < TILE_SIZE / VEC_SIZE; ++i) {
		cur[ (i*t+ty) + TILE_SIZE * tx] = A[aBegin + K*(i*t+ty) + tx];
	}
	__syncthreads();

	for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
		// Load the next submatrix to another register files.
		// Should check the out-of-range indexing to avoid kernel crash.
		if (a+aStep <= aEnd) {
		    for (int i = 0; i < TILE_SIZE / VEC_SIZE; ++i) {
				nxt[ (i*t)+ty + TILE_SIZE * tx] = A[a + K*(i*t+ty) + tx + aStep];
			}
		}
		int *ap = cur;
		int *bp = &B[b + TILE_SIZE * ty + tx];

		for (int i = 0; i < TILE_SIZE; ++i) {
			int bv = *bp;
			for (int j = 0; j < TILE_SIZE; ++j) {
				cv[j] += ap[j] * bv;
			}
			ap += TILE_SIZE;
			bp += N;
		}
		__syncthreads();

		// Swap current submatrix and next submatrix.
		// Note that you can't directly assign nxt to cur, which
		// will change cur and nxt simultaneously at the next loop.
		int *tmp = cur;
		cur = nxt;
		nxt = tmp;
	}

	int c = N * TILE_SIZE * by + TILE_SIZE * VEC_SIZE * bx;
	c += TILE_SIZE * ty + tx;
	for (int i = 0; i < TILE_SIZE; ++i) {
		C[c] = cv[i];
		c += N;
	}
}
