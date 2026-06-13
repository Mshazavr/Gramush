#include <iostream>
#include <cuda_runtime_api.h>
#include <memory.h>
#include <assert.h>
#include "helpers.hpp"

#define MAX_BLOCK_THREADS 1024

#define C_BLOCK_TILE_ROWS 64//128
#define C_BLOCK_TILE_COLS 64//128 
#define SMEM_BATCH_LENGTH 32 
#define C_THREAD_TILE_ROWS 4
#define C_THREAD_TILE_COLS 4


// Kernel definition
__global__ void mmul(float* A, float* B, float* C, int M, int K, int N)
{
    // C col: blockIdx.x * C_BLOCK_TILE_COLS
    // C row: blockIdx.y * C_BLOCK_TILE_ROWS
    A += blockIdx.y * C_BLOCK_TILE_ROWS * K;
    B += blockIdx.x * C_BLOCK_TILE_COLS;
    C += blockIdx.y * C_BLOCK_TILE_ROWS * N + blockIdx.x * C_BLOCK_TILE_COLS;

    __shared__ float sA[C_BLOCK_TILE_ROWS][SMEM_BATCH_LENGTH];
    __shared__ float sB[SMEM_BATCH_LENGTH][C_BLOCK_TILE_COLS];

    float tC[C_THREAD_TILE_ROWS][C_THREAD_TILE_COLS] = {0.0};

    for (int batch_start = 0; batch_start < K; batch_start += SMEM_BATCH_LENGTH) {

        for (int row_offset = 0; row_offset < C_BLOCK_TILE_ROWS; row_offset += blockDim.y) {
            for (int col_offset = 0; col_offset < SMEM_BATCH_LENGTH; col_offset += blockDim.x) {
                if (
                    row_offset + threadIdx.y + blockIdx.y * C_BLOCK_TILE_ROWS < M &&
                    col_offset + threadIdx.x + batch_start < K
                ){
                    sA[row_offset + threadIdx.y][col_offset + threadIdx.x] = A[(row_offset + threadIdx.y) * K + col_offset + threadIdx.x];
                }
                else {
                    sA[row_offset + threadIdx.y][col_offset + threadIdx.x] = 0.0;
                }
            }
        }

        for (int row_offset = 0; row_offset < SMEM_BATCH_LENGTH; row_offset += blockDim.y) {
            for (int col_offset = 0; col_offset < C_BLOCK_TILE_COLS; col_offset += blockDim.x) {

                if (
                    row_offset + threadIdx.y + batch_start < K &&
                    col_offset + threadIdx.x + blockIdx.x * C_BLOCK_TILE_COLS < N
                ) {
                    sB[row_offset + threadIdx.y][col_offset + threadIdx.x] = B[(row_offset + threadIdx.y) * N + col_offset + threadIdx.x];
                }
                else {
                    sB[row_offset + threadIdx.y][col_offset + threadIdx.x] = 0.0;
                }
            }
        }

        __syncthreads();
        A += SMEM_BATCH_LENGTH;
        B += SMEM_BATCH_LENGTH * N;

        for (int row = 0; row < C_THREAD_TILE_ROWS; ++row) {
            for (int col = 0; col < C_THREAD_TILE_COLS; ++col) {
                for (int i = 0; i < SMEM_BATCH_LENGTH; ++i) {
                    tC[row][col] += sA[threadIdx.y * C_THREAD_TILE_ROWS + row][i] * sB[i][threadIdx.x * C_THREAD_TILE_COLS + col];
                }
            }
        }

        __syncthreads();
    }

    for (int row = 0; row < C_THREAD_TILE_ROWS; ++row) {
        for (int col = 0; col < C_THREAD_TILE_COLS; ++col) {
            if (
                threadIdx.y * C_THREAD_TILE_ROWS + row + blockIdx.y * C_BLOCK_TILE_ROWS < M && 
                threadIdx.x * C_THREAD_TILE_COLS + col + blockIdx.x * C_BLOCK_TILE_COLS < N
            ) {
                C[(threadIdx.y * C_THREAD_TILE_ROWS + row) * N + threadIdx.x * C_THREAD_TILE_COLS + col] = tC[row][col];
            }
        }
    }
}


void cuda_mmul(float *A, float *B, float *C, int M, int K, int N) {
    int gridDimX = CEIL_DIV(N, C_BLOCK_TILE_COLS);
    int gridDimY = CEIL_DIV(M, C_BLOCK_TILE_ROWS);

    assert((C_BLOCK_TILE_COLS / C_THREAD_TILE_COLS) * (C_BLOCK_TILE_ROWS / C_THREAD_TILE_ROWS) <= MAX_BLOCK_THREADS);
    assert(C_BLOCK_TILE_COLS % C_THREAD_TILE_COLS == 0 && C_BLOCK_TILE_ROWS % C_THREAD_TILE_ROWS == 0);
    dim3 blockDim(C_BLOCK_TILE_COLS / C_THREAD_TILE_COLS, C_BLOCK_TILE_ROWS / C_THREAD_TILE_ROWS);
    dim3 gridDim(gridDimX, gridDimY);

    mmul<<<gridDim, blockDim>>>(A, B, C, M, K, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}