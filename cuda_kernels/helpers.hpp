#pragma once

#include <cuda_runtime.h>

#define CUDA_CHECK(expr_to_check) do {            \
    cudaError_t result  = expr_to_check;          \
    if(result != cudaSuccess)                     \
    {                                             \
        fprintf(stderr,                           \
                "CUDA Runtime Error: %s:%i:%d = %s\n", \
                __FILE__,                         \
                __LINE__,                         \
                result,\
                cudaGetErrorString(result));      \
    }                                             \
} while(0)


#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

inline bool isCudaDevicePointer(const void* ptr) {
    cudaPointerAttributes attributes;
    cudaError_t err = cudaPointerGetAttributes(&attributes, ptr);
    
    if (err != cudaSuccess) {
        cudaGetLastError(); 
        return false;
    }
    
    return (attributes.type == cudaMemoryTypeDevice);
}