#include <iostream>
#include <cuda_runtime_api.h>
#include <memory.h>

// Kernel definition
__global__ void vecAdd(float* A, float* B, float* C, int vectorLength)
{
    int workIndex = threadIdx.x + blockDim.x * blockIdx.x; 
    if (workIndex < vectorLength) {
        C[workIndex] = A[workIndex] + B[workIndex];
    }
}

void kernel_launcher() {
    float A[1025];
    float B[1025];
    float C[1025];
    int vectorLength = 1025;
    for (size_t i = 0; i < vectorLength; ++i) {
        A[i] = i;
        B[i] = 10 * i;
        C[i] = 0;
    }

    int blockDim = 256;
    int gridDim = (vectorLength + blockDim - 1) / blockDim;

    float *gA, *gB, *gC;
    cudaMalloc(&gA, sizeof(float)*vectorLength);
    cudaMalloc(&gB, sizeof(float)*vectorLength);
    cudaMalloc(&gC, sizeof(float)*vectorLength);

    cudaMemcpy(gA, A, sizeof(float)*vectorLength, cudaMemcpyHostToDevice);
    cudaMemcpy(gB, B, sizeof(float)*vectorLength, cudaMemcpyHostToDevice);

    vecAdd<<<gridDim, blockDim>>>(gA, gB, gC, vectorLength);

    cudaMemcpy(C, gC, sizeof(float)*vectorLength, cudaMemcpyDeviceToHost);

    for(size_t i = 0; i < vectorLength; ++i) {
        std::cout << C[i] << " ";
    }
    std::cout << std::endl;
}