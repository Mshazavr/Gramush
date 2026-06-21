#include <iostream>
#include <cuda_runtime_api.h>
#include <memory.h>
#include <assert.h>
#include <cuda/atomic>
#include "helpers.hpp"

#define BLOCK_SIZE 256

__global__ void reduce_add(float *A, float *B, int M, int N, float scale) {
    int vec_index = blockIdx.x * blockDim.x + threadIdx.x;
    int work_index = blockIdx.y * N + blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ cuda::atomic<float, cuda::thread_scope_block> block_sum;
    if(threadIdx.x == 0) {
        block_sum.store(0.0, cuda::memory_order_relaxed);
    }
    __syncthreads();

    if (vec_index < N) {
        block_sum.fetch_add(A[work_index], cuda::memory_order_relaxed);
    }
    
    __syncthreads();

    cuda::atomic_ref<float, cuda::thread_scope_device> batch_sum(B[blockIdx.y]);

    if (threadIdx.x == 0) {
        batch_sum.fetch_add(block_sum * scale, cuda::memory_order_relaxed);
    }
}


void cuda_reduce_add(float *A, float *B, int M, int N, float scale) {
    dim3 gridDim(CEIL_DIV(N, BLOCK_SIZE), M);
    reduce_add<<<gridDim, BLOCK_SIZE>>>(A, B, M, N, scale);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}


__global__ void reduce_add_backward(float *A, float *B, int M, int N, float scale) {
    int vec_index = blockIdx.x * blockDim.x + threadIdx.x;
    int work_index = blockIdx.y * N + blockIdx.x * blockDim.x + threadIdx.x;

    if (vec_index < N) {
        A[work_index] += B[blockIdx.y] * scale;
    }
}


void cuda_reduce_add_backward(float *A, float *B, int M, int N, float scale) {
    dim3 gridDim(CEIL_DIV(N, BLOCK_SIZE), M);
    reduce_add_backward<<<gridDim, BLOCK_SIZE>>>(A, B, M, N, scale);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}


#define THREAD_TILE_HEIGHT 32

__global__ void reduce_add_vertical(float *A, float *B, int M, int N, float scale) {
    A += blockIdx.y * THREAD_TILE_HEIGHT * N;
    int vec_idx = blockIdx.x * BLOCK_SIZE + threadIdx.x; 

    if (vec_idx >= N) {
        return;
    }
    
    float thread_sum = 0;
    
    for (int ind = 0; ind < THREAD_TILE_HEIGHT && ind + blockIdx.y * THREAD_TILE_HEIGHT < M; ++ind) {
        thread_sum += A[ind * N + vec_idx];
    }
    thread_sum *= scale;

    cuda::atomic_ref<float, cuda::thread_scope_device> sum_dest(B[vec_idx]);
    sum_dest.fetch_add(thread_sum, cuda::memory_order_relaxed);
}


void cuda_reduce_add_vertical(float *A, float *B, int M, int N, float scale) {
    dim3 gridDim(CEIL_DIV(N, BLOCK_SIZE), CEIL_DIV(M, THREAD_TILE_HEIGHT));
    reduce_add_vertical<<<gridDim, BLOCK_SIZE>>>(A, B, M, N, scale);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}