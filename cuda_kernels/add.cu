#include <iostream>
#include <cuda_runtime_api.h>
#include <memory.h>
#include <assert.h>
#include "helpers.hpp"

#define BLOCK_SIZE 256

__global__ void add(float *A, float *B, float *C, int N) {
    int work_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (work_index < N) {
        C[work_index] = A[work_index] + B[work_index];
    }
}

void cuda_add(float *A, float *B, float *C, int N) {
    float *gA, *gB, *gC;
    CUDA_CHECK(cudaMalloc(&gA, sizeof(float)*N));
    CUDA_CHECK(cudaMalloc(&gB, sizeof(float)*N));
    CUDA_CHECK(cudaMalloc(&gC, sizeof(float)*N));

    CUDA_CHECK(cudaMemcpy(gA, A, sizeof(float)*N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gB, B, sizeof(float)*N, cudaMemcpyHostToDevice));

    int gridDim = CEIL_DIV(N, BLOCK_SIZE);
    add<<<gridDim, BLOCK_SIZE>>>(gA, gB, gC, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(C, gC, sizeof(float)*N, cudaMemcpyDeviceToHost));
}