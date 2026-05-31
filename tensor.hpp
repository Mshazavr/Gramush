#pragma once 

#include <vector>
#include "arena.hpp"

struct Tensor; 

using TensorHandle = Tensor*;

// dims need to belong to the same arena as allocator
TensorHandle tensor_zeroes(size_t num_dims, size_t *dims, bool learnable, ArenaAllocatorHandle allocator);
TensorHandle tensor_zeroes(std::initializer_list<size_t> dims, bool learnable, ArenaAllocatorHandle allocator);

// dims need to belong to the same arena as allocator
TensorHandle tensor_random(size_t num_dims, size_t *dims, bool learnable, ArenaAllocatorHandle allocator);
TensorHandle tensor_random(std::initializer_list<size_t> dims, bool learnable, ArenaAllocatorHandle allocator);

TensorHandle tensor_from_vector(std::initializer_list<size_t> dims, bool learnable, const std::vector<float> &data, ArenaAllocatorHandle allocator);
TensorHandle tensor_from_data(std::initializer_list<size_t> dims, bool learnable, const float* data, ArenaAllocatorHandle allocator);

float tensor_at(TensorHandle tensor, const std::vector<size_t> &indices);

size_t tensor_num_sub_tensors(TensorHandle tensor, size_t num_last_dims);

size_t tensor_size(TensorHandle tensor);