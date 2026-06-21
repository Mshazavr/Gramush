#pragma once

#include <cuda_runtime.h>

#define CUDA_CHECK(expr_to_check) do {                         \
    cudaError_t __cuda_check_macro_result  = expr_to_check;    \
    if(__cuda_check_macro_result != cudaSuccess)               \
    {                                                          \
        fprintf(stderr,                                        \
                "CUDA Runtime Error: %s:%i:%d = %s\n",         \
                __FILE__,                                      \
                __LINE__,                                      \
                __cuda_check_macro_result,                     \
                cudaGetErrorString(__cuda_check_macro_result));\
    }                                                          \
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