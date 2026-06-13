#include <iostream>
#include <cuda_runtime_api.h>
#include <memory.h>
#include <assert.h>
#include "helpers.hpp"

#define BLOCK_SIZE 256

__global__ void add(float *A, float *B, float *C, int N, float alpha, float beta) {
    int work_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (work_index < N) {
        C[work_index] = alpha * A[work_index] + beta * B[work_index];
    }
}

void cuda_add(float *A, float *B, float *C, int N, float alpha, float beta) {
    int gridDim = CEIL_DIV(N, BLOCK_SIZE);
    add<<<gridDim, BLOCK_SIZE>>>(A, B, C, N, alpha, beta);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}