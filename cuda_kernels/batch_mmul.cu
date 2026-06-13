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


template<bool TransA, bool TransB>
__global__ void batch_mmul(float* A, float* B, float* C, int batch_size, int M, int K, int N, float alpha, float beta)
{
    // C col: blockIdx.x * C_BLOCK_TILE_COLS
    // C row: blockIdx.y * C_BLOCK_TILE_ROWS
    A += (!TransA ? blockIdx.y * C_BLOCK_TILE_ROWS * K : blockIdx.y * C_BLOCK_TILE_ROWS) + blockIdx.z * M * K;
    B += (!TransB ? blockIdx.x * C_BLOCK_TILE_COLS : blockIdx.x * C_BLOCK_TILE_COLS * K) + blockIdx.z * K * N;
    C += blockIdx.y * C_BLOCK_TILE_ROWS * N + blockIdx.x * C_BLOCK_TILE_COLS             + blockIdx.z * M * N;

    __shared__ float sA[C_BLOCK_TILE_ROWS][SMEM_BATCH_LENGTH];
    __shared__ float sB[SMEM_BATCH_LENGTH][C_BLOCK_TILE_COLS];

    float tC[C_THREAD_TILE_ROWS][C_THREAD_TILE_COLS] = {0.0};

    for (int batch_start = 0; batch_start < K; batch_start += SMEM_BATCH_LENGTH) {

        if (!TransA) {
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
        }
        else {
        for (int col_offset = 0; col_offset < SMEM_BATCH_LENGTH; col_offset += blockDim.y) {
            for (int row_offset = 0; row_offset < C_BLOCK_TILE_ROWS; row_offset += blockDim.x) {
                if (
                    row_offset + threadIdx.x + blockIdx.y * C_BLOCK_TILE_ROWS < M &&
                    col_offset + threadIdx.y + batch_start < K
                ){
                    sA[row_offset + threadIdx.x][col_offset + threadIdx.y] = A[row_offset + threadIdx.x + (col_offset + threadIdx.y) * M];
                }
                else {
                    sA[row_offset + threadIdx.x][col_offset + threadIdx.y] = 0.0;
                }
            }
        }
        }

        if (!TransB) {
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
        }
        else {
        for (int col_offset = 0; col_offset < C_BLOCK_TILE_COLS; col_offset += blockDim.y) {
            for (int row_offset = 0; row_offset < SMEM_BATCH_LENGTH; row_offset += blockDim.x) {
                if (
                    row_offset + threadIdx.x + batch_start < K &&
                    col_offset + threadIdx.y + blockIdx.x * C_BLOCK_TILE_COLS < N
                ) {
                    sB[row_offset + threadIdx.x][col_offset + threadIdx.y] = B[(row_offset + threadIdx.x) + (col_offset + threadIdx.y) * K];
                }
                else {
                    sB[row_offset + threadIdx.x][col_offset + threadIdx.y] = 0.0;
                }
            }
        }
        }

        __syncthreads();
        A += (!TransA) ? SMEM_BATCH_LENGTH : SMEM_BATCH_LENGTH * M;
        B += (!TransB) ? SMEM_BATCH_LENGTH * N : SMEM_BATCH_LENGTH;

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
                C[(threadIdx.y * C_THREAD_TILE_ROWS + row) * N + threadIdx.x * C_THREAD_TILE_COLS + col] = (
                    alpha * tC[row][col] + beta * C[(threadIdx.y * C_THREAD_TILE_ROWS + row) * N + threadIdx.x * C_THREAD_TILE_COLS + col]
                );
            }
        }
    }
}

// TODO: the parameters already should be device pointers
void cuda_batch_mmul(float *A, bool trans_a, float *B, bool trans_b, float *C, int batch_size, int M, int K, int N, float alpha, float beta) {
    int gridDimX = CEIL_DIV(N, C_BLOCK_TILE_COLS);
    int gridDimY = CEIL_DIV(M, C_BLOCK_TILE_ROWS);

    assert((C_BLOCK_TILE_COLS / C_THREAD_TILE_COLS) * (C_BLOCK_TILE_ROWS / C_THREAD_TILE_ROWS) <= MAX_BLOCK_THREADS);
    assert(C_BLOCK_TILE_COLS % C_THREAD_TILE_COLS == 0 && C_BLOCK_TILE_ROWS % C_THREAD_TILE_ROWS == 0);
    dim3 blockDim(C_BLOCK_TILE_COLS / C_THREAD_TILE_COLS, C_BLOCK_TILE_ROWS / C_THREAD_TILE_ROWS);
    dim3 gridDim(gridDimX, gridDimY, batch_size);

    switch ((int)trans_a + 2 * (int)trans_b) {
        case 0b00:
            batch_mmul<false, false><<<gridDim, blockDim>>>(A, B, C, batch_size, M, K, N, alpha, beta);
            break;
        case 0b01:
            batch_mmul<true, false><<<gridDim, blockDim>>>(A, B, C, batch_size, M, K, N, alpha, beta);
            break;
        case 0b10:
            batch_mmul<false, true><<<gridDim, blockDim>>>(A, B, C, batch_size, M, K, N, alpha, beta);
            break;
        case 0b11:
            batch_mmul<true, true><<<gridDim, blockDim>>>(A, B, C, batch_size, M, K, N, alpha, beta);
            break;
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}