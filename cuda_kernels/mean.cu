#include <iostream>
#include <cuda_runtime_api.h>
#include <memory.h>
#include <assert.h>
#include "helpers.hpp"

#define BLOCK_SIZE 256

__global__ void batch_mean(float *A, float *B, float *sum, float *count, int M, int N) {
    int batch_num = blockIdx.y;
    int work_index = blockIdx.y * M + blockIdx.x * blockDim.x + threadIdx.x;
    if (work_index < N) {
        sum[batch_num] += A[work_index];
        count[batch_num] += 1;
    }
    
    __syncthreads();
    if (blockIdx.x * blockDim.x + threadIdx.x == 0)
}


void cuda_batch_mean(float *A, float *B, int M, int N) {
    float *gA, *gB, *sum, *count;
    CUDA_CHECK(cudaMalloc(&gA, sizeof(float)*M*N));
    CUDA_CHECK(cudaMalloc(&gB, sizeof(float)*M));
    CUDA_CHECK(cudaMalloc(&sum, sizeof(float)*M));
    CUDA_CHECK(cudaMalloc(&count, sizeof(float)*M));



    CUDA_CHECK(cudaMemcpy(gA, A, sizeof(float)*M*N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gB, B, sizeof(float)*M, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(sum, 0.0, sizeof(float)*M));
    CUDA_CHECK(cudaMemset(count, 0.0, sizeof(float)*M));

    dim3 gridDim(CEIL_DIV(N, BLOCK_SIZE), M);
    batch_mean<<<gridDim, BLOCK_SIZE>>>(gA, gB, sum, count, M, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(B, gB, sizeof(float)*M, cudaMemcpyDeviceToHost));
}