#include <stdio.h>

struct ArenaAllocator;

using ArenaAllocatorHandle = ArenaAllocator*;

ArenaAllocatorHandle arena_init(size_t n_bytes, size_t cuda_n_bytes);
void *arena_alloc(ArenaAllocatorHandle allocator, size_t n_bytes, size_t alignment);
void *arena_cuda_alloc(ArenaAllocatorHandle allocator, size_t n_bytes, size_t alignment);
void arena_reset(ArenaAllocatorHandle allocator);
void arena_free(ArenaAllocatorHandle allocator);