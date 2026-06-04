#include "arena.hpp"
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <cstring>
#include <cmath>

struct ArenaAllocator {
    char *buffer;
    char *next;
    size_t size;
};

ArenaAllocatorHandle arena_init(size_t n_bytes) {
    ArenaAllocatorHandle arena = (ArenaAllocatorHandle)malloc(sizeof(ArenaAllocator));
    arena->buffer = (char*)calloc(n_bytes, 1);
    arena->size = n_bytes;
    arena->next = arena->buffer;
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

void arena_reset(ArenaAllocatorHandle allocator) {
    memset(allocator->buffer, 0, sizeof(char*) * std::min(uintptr_t(allocator->next - allocator->buffer), allocator->size));
    allocator->next = allocator->buffer;
}

void arena_free(ArenaAllocatorHandle allocator) {
    if (allocator) {
        free(allocator->buffer);
        free(allocator);
    }
}