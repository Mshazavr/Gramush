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
    float *gA, *gB;
    CUDA_CHECK(cudaMalloc(&gA, sizeof(float)*N));
    CUDA_CHECK(cudaMalloc(&gB, sizeof(float)*N));
    CUDA_CHECK(cudaMemcpy(gA, A, sizeof(float)*N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gB, B, sizeof(float)*N, cudaMemcpyHostToDevice));

    int gridDim = CEIL_DIV(N, BLOCK_SIZE);
    relu<<<gridDim, BLOCK_SIZE>>>(gA, gB, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(B, gB, sizeof(float)*N, cudaMemcpyDeviceToHost));
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
    float *gA, *gAG, *gBG;
    CUDA_CHECK(cudaMalloc(&gA, sizeof(float)*N));
    CUDA_CHECK(cudaMalloc(&gAG, sizeof(float)*N));
    CUDA_CHECK(cudaMalloc(&gBG, sizeof(float)*N));
    
    CUDA_CHECK(cudaMemcpy(gA, A, sizeof(float)*N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gAG, AG, sizeof(float)*N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gBG, BG, sizeof(float)*N, cudaMemcpyHostToDevice));

    int gridDim = CEIL_DIV(N, BLOCK_SIZE);
    relu_backward<<<gridDim, BLOCK_SIZE>>>(gA, gAG, gBG, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(AG, gAG, sizeof(float)*N, cudaMemcpyDeviceToHost));
}