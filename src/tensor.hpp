#pragma once 

#include <vector>
#include "arena.hpp"

enum class DType {
    FLOAT32,
    SIZE_T
};

enum class MType {
    HOST,
    CUDA_DEVICE
};

struct Tensor; 

using TensorHandle = Tensor*;

// dims need to belong to the same arena as allocator
TensorHandle tensor_zeroes(MType mtype, DType dtype, size_t num_dims, size_t *dims, bool learnable, ArenaAllocatorHandle allocator);
TensorHandle tensor_zeroes(MType mtype, DType dtype, std::initializer_list<size_t> dims, bool learnable, ArenaAllocatorHandle allocator);

// dims need to belong to the same arena as allocator
TensorHandle tensor_random(MType mtype, DType dtype, size_t num_dims, size_t *dims, bool learnable, ArenaAllocatorHandle allocator);
TensorHandle tensor_random(MType mtype, DType dtype, std::initializer_list<size_t> dims, bool learnable, ArenaAllocatorHandle allocator);

TensorHandle tensor_from_vector(MType mtype, DType dtype, std::initializer_list<size_t> dims, bool learnable, const std::vector<float> &data, ArenaAllocatorHandle allocator);
TensorHandle tensor_from_data(MType mtype, DType dtype, std::initializer_list<size_t> dims, bool learnable, const float* data, ArenaAllocatorHandle allocator);

float tensor_at(TensorHandle tensor, const std::vector<size_t> &indices);

size_t tensor_num_sub_tensors(TensorHandle tensor, size_t num_last_dims);

size_t tensor_size(TensorHandle tensor);