#include <iostream>
#include <cuda_runtime_api.h>
#include <memory.h>
#include <assert.h>
#include <cmath>
#include "helpers.hpp"

#define BLOCK_SIZE 256

__global__ void relu(float *A, float *B, int N) {
    int work_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (work_index < N) {
        B[work_index] = fmaxf(A[work_index], 0.0f);
    }
}

void cuda_relu(float *A, float *B, int N) {
    int gridDim = CEIL_DIV(N, BLOCK_SIZE);
    relu<<<gridDim, BLOCK_SIZE>>>(A, B, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}


__global__ void relu_backward(float *A, float *AG, float *BG, int N) {
    int work_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (work_index < N) {
        if (A[work_index] >= 0) {
            AG[work_index] += BG[work_index];
        }
    }
}

void cuda_relu_backward(float *A, float *AG, float *BG, int N) {
    int gridDim = CEIL_DIV(N, BLOCK_SIZE);
    relu_backward<<<gridDim, BLOCK_SIZE>>>(A, AG, BG, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}