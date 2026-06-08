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
        batch_sum.fetch_add(block_sum, cuda::memory_order_relaxed);
    }

    __syncthreads();

    if (work_index < M) {
        B[work_index] = B[work_index] * scale;
    }
}


void cuda_reduce_add(float *A, float *B, int M, int N, float scale) {
    float *gA, *gB;
    CUDA_CHECK(cudaMalloc(&gA, sizeof(float)*M*N));
    CUDA_CHECK(cudaMalloc(&gB, sizeof(float)*M));

    CUDA_CHECK(cudaMemcpy(gA, A, sizeof(float)*M*N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gB, B, sizeof(float)*M, cudaMemcpyHostToDevice));

    dim3 gridDim(CEIL_DIV(N, BLOCK_SIZE), M);
    reduce_add<<<gridDim, BLOCK_SIZE>>>(gA, gB, M, N, scale);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(B, gB, sizeof(float)*M, cudaMemcpyDeviceToHost));
}


__global__ void reduce_add_backward(float *A, float *B, int M, int N, float scale) {
    int vec_index = blockIdx.x * blockDim.x + threadIdx.x;
    int work_index = blockIdx.y * N + blockIdx.x * blockDim.x + threadIdx.x;

    if (vec_index < N) {
        A[work_index] += B[blockIdx.y] * scale;
    }
}


void cuda_reduce_add_backward(float *A, float *B, int M, int N, float scale) {
    float *gA, *gB;
    CUDA_CHECK(cudaMalloc(&gA, sizeof(float)*M*N));
    CUDA_CHECK(cudaMalloc(&gB, sizeof(float)*M));

    CUDA_CHECK(cudaMemcpy(gA, A, sizeof(float)*M*N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gB, B, sizeof(float)*M, cudaMemcpyHostToDevice));

    dim3 gridDim(CEIL_DIV(N, BLOCK_SIZE), M);
    reduce_add_backward<<<gridDim, BLOCK_SIZE>>>(gA, gB, M, N, scale);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(A, gA, sizeof(float)*M*N, cudaMemcpyDeviceToHost));
}