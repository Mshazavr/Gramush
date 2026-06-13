#include <iostream>
#include <cuda_runtime_api.h>
#include <memory.h>
#include <assert.h>
#include "helpers.hpp"
#include "logsoftmax.hpp"
#include <cuda/atomic>
#include <cfloat>

#define BLOCK_SIZE 256

__global__ void logsoftmax(float *A, float *B, float *C, float* row_exp_sum, float *row_max, int batch_size, int vec_size) {    
    // compute the max for current batch index and store in row_max[blockIdx.y]
    // ------------------------------------------------------------------------

    A += blockIdx.y * vec_size;
    B += blockIdx.y * vec_size;

    float thread_max = -FLT_MAX;
    for (int ind = threadIdx.x; ind < vec_size; ind += blockDim.x) {
        thread_max = fmaxf(thread_max, A[ind]);
    }

    __shared__ cuda::atomic<float, cuda::thread_scope_block> block_max; 
    if (threadIdx.x == 0) {
        block_max.store(thread_max, cuda::memory_order_relaxed);
    }
    __syncthreads();
    
    block_max.fetch_max(thread_max, cuda::memory_order_relaxed);
    __syncthreads();

    float final_block_max = block_max.load(cuda::memory_order_relaxed);

    cuda::atomic_ref<float, cuda::thread_scope_device> batch_row_max(row_max[blockIdx.y]);
    if (threadIdx.x == 0) {
        batch_row_max.store(final_block_max, cuda::memory_order_relaxed);
    }
    __syncthreads();


    // compute the exp sum for current batch index and store in exp_sum[blockIdx.y]
    // ----------------------------------------------------------------------------
    float thread_exp_sum = 0.0;
    for (int ind = threadIdx.x; ind < vec_size; ind += blockDim.x) {
        thread_exp_sum += __expf(A[ind] - final_block_max);
    }

    __shared__ cuda::atomic<float, cuda::thread_scope_block> block_exp_sum; 
    if (threadIdx.x == 0) {
        block_exp_sum.store(0, cuda::memory_order_relaxed);
    }
    __syncthreads();
    
    block_exp_sum.fetch_add(thread_exp_sum, cuda::memory_order_relaxed);
    __syncthreads();

    float final_block_exp_sum = block_exp_sum.load(cuda::memory_order_relaxed);

    cuda::atomic_ref<float, cuda::thread_scope_device> batch_row_exp_sum(row_exp_sum[blockIdx.y]);
    if (threadIdx.x == 0) {
        batch_row_exp_sum.store(final_block_exp_sum, cuda::memory_order_relaxed);
    }
    __syncthreads();

    
    // Compute the final cross-entropy loss 
    // ------------------------------------

    for (int ind = threadIdx.x; ind < vec_size; ind += blockDim.x) {
        if (B[ind] == 1.0f) {
            C[blockIdx.y] = -A[ind] + final_block_max + __logf(final_block_exp_sum);
        }
    }

}

void cuda_logsoftmax(float *A, float *B, float *C, float *row_exp_sum, float *row_max, int batch_size, int vec_size) {
    dim3 gridDim(1, batch_size);
    logsoftmax<<<gridDim, BLOCK_SIZE>>>(A, B, C, row_exp_sum, row_max, batch_size, vec_size);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}


__global__ void logsoftmax_backward(float *A, float *A_grads, float *B, float *C_grads, float *row_exp_sum, float *row_max, int batch_size, int vec_size) {
    int vec_index = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int work_index = blockIdx.y * vec_size + blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (vec_index < vec_size) {
        float c_grad = C_grads[blockIdx.y];
        A_grads[work_index] +=c_grad * (__expf(A[work_index] - row_max[blockIdx.y]) / row_exp_sum[blockIdx.y]);
        if (B[work_index] == 1.0f) {
            A_grads[work_index] -= c_grad;
        }
    }
}


void cuda_logsoftmax_backward(float *A, float *A_grads, float *B, float *C_grads, float *row_exp_sum, float *row_max, int batch_size, int vec_size) {
    dim3 gridDim(CEIL_DIV(vec_size, BLOCK_SIZE), batch_size);
    logsoftmax_backward<<<gridDim, BLOCK_SIZE>>>(A, A_grads, B, C_grads, row_exp_sum, row_max, batch_size, vec_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}