#include "arena.hpp"
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <cstring>
#include <cmath>
#include <cuda_runtime_api.h>
#include "cuda_kernels/helpers.hpp"

struct ArenaAllocator {
    char *buffer;
    char *next;
    size_t size;

    char *cuda_buffer;
    char *cuda_next;
    size_t cuda_size;
};

ArenaAllocatorHandle arena_init(size_t n_bytes, size_t cuda_n_bytes) {
    ArenaAllocatorHandle arena = (ArenaAllocatorHandle)malloc(sizeof(ArenaAllocator));
    arena->buffer = (char*)calloc(n_bytes, 1);
    arena->size = n_bytes;
    arena->next = arena->buffer;

    CUDA_CHECK(cudaMalloc(&arena->cuda_buffer, cuda_n_bytes));
    CUDA_CHECK(cudaMemset(arena->cuda_buffer, 0, cuda_n_bytes));
    arena->cuda_size = cuda_n_bytes;
    arena->cuda_next = arena->cuda_buffer;
    return arena;
}

void *arena_alloc(ArenaAllocatorHandle allocator, size_t n_bytes, size_t alignment) {
    uintptr_t next_num = (uintptr_t)allocator->next; 
    next_num = (next_num + alignment - 1) & ~(alignment - 1);
    allocator->next = (char*)next_num;
    
    if (allocator->next + n_bytes > allocator->buffer + allocator->size) {
        // TODO: fix this
        std::cerr << "Run out of arena memory!" << std::endl; 
        exit(0);
    }

    void *result = allocator->next;
    allocator->next += n_bytes;
    return result;
}

void *arena_cuda_alloc(ArenaAllocatorHandle allocator, size_t n_bytes, size_t alignment) {
    uintptr_t next_num = (uintptr_t)allocator->cuda_next; 
    next_num = (next_num + alignment - 1) & ~(alignment - 1);
    allocator->cuda_next = (char*)next_num;
    
    if (allocator->cuda_next + n_bytes > allocator->cuda_buffer + allocator->cuda_size) {
        // TODO: fix this
        std::cerr << "Run out of arena cuda memory!" << std::endl; 
        exit(0);
    }

    void *result = allocator->cuda_next;
    allocator->cuda_next += n_bytes;
    return result;
}

void arena_reset(ArenaAllocatorHandle allocator) {
    memset(
        allocator->buffer, 
        0, 
        sizeof(char) * std::min(uintptr_t(allocator->next - allocator->buffer), allocator->size)
    );
    allocator->next = allocator->buffer;

    cudaMemset(
        allocator->cuda_buffer, 
        0, 
        std::min(uintptr_t(allocator->cuda_next - allocator->cuda_buffer), allocator->cuda_size)
    );
    allocator->cuda_next = allocator->cuda_buffer;
}

void arena_free(ArenaAllocatorHandle allocator) {
    if (allocator) {
        free(allocator->buffer);
        cudaFree(allocator->cuda_buffer);
        free(allocator);
    }
}