//////////////////////////////////////////////////////////////////////
// A simple example to show how CUDA WMMA API works with Tensor Cores
//    Created by Zong-Sheng Wang @ 2018/11/25
// Performance Tips:
//    To minimize bank conflicts, you should try to shift row or 
// column of matrics in shared memory
// cmd: 
//    $ nvcc -o main main.cu -arch sm_75

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <mma.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define WARP_SIZE 32

// MMA matrix tile dimensions.
#define M 16
#define N 16
#define K 16

// GEMM configuration.
#define M_TILES 1
#define N_TILES 1
#define K_TILES 1

#define M_TOTAL (M * M_TILES)
#define N_TOTAL (N * N_TILES)
#define K_TOTAL (K * K_TILES)


//__global__ void WMMAINT8()
using namespace nvcuda;

__host__ void InitMatrix(half *A, half *B, half *C)
{
	for (int i = 0; i < M_TOTAL*K_TOTAL; i++)
		A[i] = i;
	for (int i = 0; i < K_TOTAL*N_TOTAL; i++)
		B[i] = i;
	for (int i = 0; i < M_TOTAL*N_TOTAL; i++)
	 	C[i] = 0;
}

__global__ void WMMAF16TensorCore(half *A, half *B, half *C) {
    int ix = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int iy = (blockIdx.y * blockDim.y + threadIdx.y);

	C[5] = 1.02 ; C[6] = 1.1;

/*
    wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, M, N, K, half> ab_frag;
    
    wmma::fill_fragment(ab_frag, 0.0f);

    int a_col, a_row, b_col, b_row;
    a_row = ix * M;
    b_row = iy * N;

    for (int k = 0; k < K_TOTAL; k += K) {
        a_col = k;
        b_col = k;

        if (a_row < M_TOTAL && a_col < K_TOTAL && b_row < K_TOTAL && b_col < N_TOTAL) {
            // Load the inputs
            wmma::load_matrix_sync(a_frag, A + a_col + a_row * M_TOTAL, M_TOTAL);
            wmma::load_matrix_sync(b_frag, B + b_row + b_col * N_TOTAL, N_TOTAL);

            // Perform the matrix multiplication
            wmma::mma_sync(ab_frag, a_frag, b_frag, ab_frag);
        }
    }

    if (a_row < M_TOTAL && b_row < N_TOTAL) {
        wmma::store_matrix_sync(C + b_row * M_TOTAL + a_row, ab_frag, M_TOTAL, wmma::mem_row_major);
    }*/
}

int main()
{
	// Matrix on device
	half *A;
	half *B;
	half *C;

	// CUDA Unified Memory 
	cudaMallocManaged((void **)&A, sizeof(half) * M_TOTAL * K_TOTAL);
	cudaMallocManaged((void **)&B, sizeof(half) * K_TOTAL * N_TOTAL);
	cudaMallocManaged((void **)&C, sizeof(half) * M_TOTAL * N_TOTAL);

	InitMatrix(A, B, C);

	dim3 gridDim, blockDim;
	// 16 warps in one block
	blockDim.x = 1 * WARP_SIZE; 
	blockDim.y = 1;

	gridDim.x = (M_TOTAL + (M * blockDim.x / WARP_SIZE - 1)) / (M * blockDim.x / WARP_SIZE);
	gridDim.y = (N_TOTAL + N * blockDim.y - 1) / (N * blockDim.y);

	WMMAF16TensorCore<<<gridDim, blockDim>>>(A, B, C);
	cudaDeviceSynchronize();
	for(int i=0; i < M_TOTAL; i++)
	{
		for(int j=0; j< N_TOTAL; j++)
		{
			printf("%f  ", __half2float(C[i*N_TOTAL+j]));
		}
		printf("\n");
	}
	
	cudaFree(A);
	cudaFree(B);
	cudaFree(C);

	return 0;
}
