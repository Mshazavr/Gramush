#include "tensor_priv.hpp"
#include <vector>
#include <cstdlib>
#include <cstring>

// dims need to belong to the same arena as allocator
TensorHandle tensor_zeroes(size_t num_dims, size_t *dims, bool learnable, ArenaAllocatorHandle allocator) {
    size_t num_elems = 1;
    for (size_t i = 0; i < num_dims; ++i) {
        num_elems *= dims[i];
    }

    TensorHandle result = (TensorHandle)arena_alloc(allocator, sizeof(Tensor), alignof(Tensor));
    result->num_dims = num_dims;
    result->dims = dims;
    result->strides = (size_t*)arena_alloc(allocator, sizeof(size_t) * num_dims, alignof(size_t));
    result->data = (float*)arena_alloc(allocator, sizeof(float) * num_elems, alignof(float));
    result->grads = (float*)arena_alloc(allocator, sizeof(float) * num_elems, alignof(float));
    result->learnable = learnable;

    for (size_t i = 0; i < result->num_dims; ++i) {
        num_elems /= result->dims[i];
        result->strides[i] = num_elems;
    }

    return result;
}

TensorHandle tensor_zeroes(std::initializer_list<size_t> dims, bool learnable, ArenaAllocatorHandle allocator) {
    size_t *result_dims = (size_t*)arena_alloc(allocator, sizeof(size_t) * dims.size(), alignof(size_t));
    memcpy(result_dims, dims.begin(), sizeof(size_t) * dims.size());
    return tensor_zeroes(dims.size(), result_dims, learnable, allocator);
}

// dims need to belong to the same arena as allocator
TensorHandle tensor_random(size_t num_dims, size_t *dims, bool learnable, ArenaAllocatorHandle allocator) {
    TensorHandle result = tensor_zeroes(num_dims, dims, learnable, allocator);    
    size_t n = tensor_size(result);
    for (size_t i = 0; i < n; ++i) {
        result->data[i] = (std::rand() / (float)RAND_MAX) * 0.2f - 0.1f;
    }
    return result;
}

TensorHandle tensor_random(std::initializer_list<size_t> dims, bool learnable, ArenaAllocatorHandle allocator) {    
    size_t *result_dims = (size_t*)arena_alloc(allocator, sizeof(size_t) * dims.size(), alignof(size_t));
    memcpy(result_dims, dims.begin(), sizeof(size_t) * dims.size());
    return tensor_random(dims.size(), result_dims, learnable, allocator);
}

TensorHandle tensor_from_vector(std::initializer_list<size_t> dims, bool learnable, const std::vector<float> &data, ArenaAllocatorHandle allocator) {
    TensorHandle result = tensor_zeroes(dims, learnable, allocator);
    memcpy(result->data, data.data(), data.size() * sizeof(float));
    return result;
}

TensorHandle tensor_from_data(std::initializer_list<size_t> dims, bool learnable, const float* data, ArenaAllocatorHandle allocator) {
    TensorHandle result = tensor_zeroes(dims, learnable, allocator);
    memcpy(result->data, data, tensor_size(result) * sizeof(float));
    return result;   
}

float tensor_at(TensorHandle tensor, const std::vector<size_t> &indices) {
    size_t ind = 0;
    for (size_t i = 0; i < tensor->num_dims; ++i) {
        ind += tensor->strides[i] * indices[i];
    }
    return tensor->data[ind];
}

size_t tensor_num_sub_tensors(TensorHandle tensor, size_t num_last_dims) {
    size_t num_sub_tensors = 1;
    for (size_t i = 0; i + num_last_dims < tensor->num_dims; ++i) {
        num_sub_tensors *= tensor->dims[i];
    }
    return num_sub_tensors;
}

size_t tensor_size(TensorHandle tensor) {
    return tensor_num_sub_tensors(tensor, 0);
}