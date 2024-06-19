//////////////////////////////////////////////////////////////////////
// A simple example to show how CUDA WMMA API works with Tensor Cores
//    Created by Zong-Sheng Wang @ 2018/11/25
// Performance Tips:
//    To minimize bank conflicts, you should try to shift row or 
// column of matrics in shared memory
// cmd: 
//    $ nvcc -o add add.cu -arch sm_70

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <mma.h>
#include <curand_kernel.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define WARP_SIZE 32

// MMA matrix tile dimensions.
#define M 16
#define N 16

// GEMM configuration.
#define M_TILES 1
#define N_TILES 1

#define M_TOTAL (M * M_TILES)
#define N_TOTAL (N * N_TILES)


//__global__ void WMMAINT8()
using namespace nvcuda;

__host__ void InitMatrix(half *A, half *B, half *C)
{
	for (int i = 0; i < M_TOTAL*N; i++)
		A[i] =  __float2half( i / M_TOTAL);
	for (int i = 0; i < M*N_TOTAL; i++)
		B[i] =  __float2half( i / M_TOTAL);
	for (int i = 0; i < M_TOTAL*N_TOTAL; i++)
	 	C[i] = 0;
}

__global__ void WMMAF16TensorCore(half *A, half *B, half *C) {
    int ix = threadIdx.x / WARP_SIZE;
    int iy = threadIdx.y;

    wmma::fragment<wmma::matrix_a, M, N, M, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, M, N, M, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, M, N, M, half> ab_frag;
    
    wmma::fill_fragment(ab_frag, 0.0f);

    int a_col, a_row, b_col, b_row;
    a_row = ix * M;
    b_row = ix * M;

	a_col = iy * N;
	b_col = iy * N;

	if (a_row < M_TOTAL && a_col <  N_TOTAL  && b_row < M_TOTAL && b_col < N_TOTAL) {
		// Load the inputs
		wmma::load_matrix_sync(a_frag, A + a_col + a_row * N_TOTAL, N_TOTAL);
		wmma::load_matrix_sync(b_frag, B + b_col + b_row * N_TOTAL, N_TOTAL);

		for (int i=0; i < a_frag.num_elements; i++)
		{
			ab_frag.x[i] = a_frag.x[i]//+b_frag.x[i];
		}

		// Perform the matrix multiplication
		//wmma::mma_sync(ab_frag, a_frag, b_frag, ab_frag);
	}


    if (a_row < M_TOTAL && b_row < N_TOTAL) {
        wmma::store_matrix_sync(C + a_row * N_TOTAL + a_col, ab_frag, N_TOTAL, wmma::mem_row_major);
    }
}

int main()
{
	// Matrix on device
	half *A;
	half *B;
	half *C;

	// CUDA Unified Memory 
	cudaMallocManaged((void **)&A, sizeof(half) * M_TOTAL * N_TOTAL);
	cudaMallocManaged((void **)&B, sizeof(half) * M_TOTAL * N_TOTAL);
	cudaMallocManaged((void **)&C, sizeof(half) * M_TOTAL * N_TOTAL);

	InitMatrix(A, B, C);

	printf("Matrix A is\n");

	for(int i=0; i < M_TOTAL; i++)
	{
		for(int j=0; j< N_TOTAL; j++)
		{
			printf("%f  ", __half2float(A[i*N_TOTAL+j]));
		}
		printf("\n");
	}
	printf("Matrix B is\n");
	for(int i=0; i < M_TOTAL; i++)
	{
		for(int j=0; j< N_TOTAL; j++)
		{
			printf("%f  ", __half2float(B[i*N_TOTAL+j]));
		}
		printf("\n");
	}

	dim3 gridDim, blockDim;
	// 16 warps in one block
	blockDim.x = N_TILES * WARP_SIZE; 
	blockDim.y = M_TILES;

	gridDim.x = 1;
	gridDim.y = 1;

	WMMAF16TensorCore<<<gridDim, blockDim>>>(A, B, C);
	cudaDeviceSynchronize();
	printf("Matrix C is\n");
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
