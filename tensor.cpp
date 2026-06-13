#include "tensor_priv.hpp"
#include <vector>
#include <cstdlib>
#include <iostream>
#include <cstring>
#include <cuda_runtime_api.h>
#include "cuda_kernels/helpers.hpp"

size_t dtype_sz(DType dtype) {
    if (dtype == DType::FLOAT32) {
        return sizeof(float);
    }
    else { // dtype == DType::SIZE_T
        return sizeof(size_t);
    } 
}

size_t dtype_al(DType dtype) {
    if (dtype == DType::FLOAT32) {
        return alignof(float);
    }
    else { // dtype == DType::SIZE_T
        return alignof(size_t);
    } 
}

// dims need to belong to the same arena as allocator
TensorHandle tensor_zeroes(MType mtype, DType dtype, size_t num_dims, size_t *dims, bool learnable, ArenaAllocatorHandle allocator) {
    size_t num_elems = 1;
    for (size_t i = 0; i < num_dims; ++i) {
        num_elems *= dims[i];
    }

    TensorHandle result = (TensorHandle)arena_alloc(allocator, sizeof(Tensor), alignof(Tensor));
    result->num_dims = num_dims;
    result->dims = dims;
    result->strides = (size_t*)arena_alloc(allocator, sizeof(size_t) * num_dims, alignof(size_t));
    result->learnable = learnable;
    result->dtype = dtype;
    result->mtype = mtype;

    if (mtype == MType::HOST) {
        result->data = arena_alloc(allocator, dtype_sz(dtype) * num_elems, dtype_al(dtype));
        result->grads = arena_alloc(allocator, dtype_sz(dtype) * num_elems, dtype_al(dtype));
    }
    else { // mtype == MType::CUDA_DEVICE
        result->data = arena_cuda_alloc(allocator, dtype_sz(dtype) * num_elems, dtype_al(dtype));
        result->grads = arena_cuda_alloc(allocator, dtype_sz(dtype) * num_elems, dtype_al(dtype));
    }

    for (size_t i = 0; i < result->num_dims; ++i) {
        num_elems /= result->dims[i];
        result->strides[i] = num_elems;
    }

    return result;
}

TensorHandle tensor_zeroes(MType mtype, DType dtype, std::initializer_list<size_t> dims, bool learnable, ArenaAllocatorHandle allocator) {
    size_t *result_dims = (size_t*)arena_alloc(allocator, sizeof(size_t) * dims.size(), alignof(size_t));
    memcpy(result_dims, dims.begin(), sizeof(size_t) * dims.size());
    return tensor_zeroes(mtype, dtype, dims.size(), result_dims, learnable, allocator);
}

// dims need to belong to the same arena as allocator
TensorHandle tensor_random(MType mtype, DType dtype, size_t num_dims, size_t *dims, bool learnable, ArenaAllocatorHandle allocator) {
    TensorHandle result = tensor_zeroes(mtype, dtype, num_dims, dims, learnable, allocator);  
    size_t n = tensor_size(result);

    void *random_data_buffer;
    if (mtype == MType::HOST) {
        random_data_buffer = result->data;
    }
    else { // mtype == MType::CUDA_DEVICE
        random_data_buffer = malloc(n * dtype_sz(dtype));
    }

    for (size_t i = 0; i < n; ++i) {
        if (dtype == DType::FLOAT32) {
            ((float*)(random_data_buffer))[i] = (std::rand() / (float)RAND_MAX) * 0.2f - 0.1f;
        }
        else { // dtype == DType::SIZE_T
            ((size_t*)(random_data_buffer))[i] = (size_t)((std::rand() / (float)RAND_MAX) * 2.0f);
        }
    }

    if (mtype == MType::CUDA_DEVICE) {
        CUDA_CHECK(cudaMemcpy(result->data, random_data_buffer, n * dtype_sz(dtype), cudaMemcpyHostToDevice));
        free(random_data_buffer);
    }

    return result;
}

TensorHandle tensor_random(MType mtype, DType dtype, std::initializer_list<size_t> dims, bool learnable, ArenaAllocatorHandle allocator) {    
    size_t *result_dims = (size_t*)arena_alloc(allocator, sizeof(size_t) * dims.size(), alignof(size_t));
    memcpy(result_dims, dims.begin(), sizeof(size_t) * dims.size());
    return tensor_random(mtype, dtype, dims.size(), result_dims, learnable, allocator);
}

TensorHandle tensor_from_vector(MType mtype, DType dtype, std::initializer_list<size_t> dims, bool learnable, const std::vector<float> &data, ArenaAllocatorHandle allocator) {
    TensorHandle result = tensor_zeroes(mtype, dtype, dims, learnable, allocator);
    memcpy(result->data, data.data(), data.size() * sizeof(float));
    if (mtype == MType::HOST) {
        memcpy(result->data, data.data(), data.size() * dtype_sz(dtype));
    }
    else { // mtype == MType::CUDA_DEVICE
        CUDA_CHECK(cudaMemcpy(result->data, data.data(), data.size() * dtype_sz(dtype), cudaMemcpyHostToDevice));
    }
    return result;
}

TensorHandle tensor_from_data(MType mtype, DType dtype, std::initializer_list<size_t> dims, bool learnable, const float* data, ArenaAllocatorHandle allocator) {
    TensorHandle result = tensor_zeroes(mtype, dtype, dims, learnable, allocator);
    if (mtype == MType::HOST) {
        memcpy(result->data, data, tensor_size(result) * dtype_sz(dtype));
    }
    else { // mtype == MType::CUDA_DEVICE
        CUDA_CHECK(cudaMemcpy(result->data, data, tensor_size(result) * dtype_sz(dtype), cudaMemcpyHostToDevice));
    }
    return result;   
}

float tensor_at(TensorHandle tensor, const std::vector<size_t> &indices) {
    size_t ind = 0;
    for (size_t i = 0; i < tensor->num_dims; ++i) {
        ind += tensor->strides[i] * indices[i];
    }
    if (tensor->dtype == DType::FLOAT32) {
        if (tensor->mtype == MType::HOST) {
            return ((float*)tensor->data)[ind];
        }
        else { // tensor->mtype == MType::CUDA_DEVICE
            float tmp;
            CUDA_CHECK(cudaMemcpy(&tmp, (float*)tensor->data + ind, dtype_sz(tensor->dtype), cudaMemcpyDeviceToHost));
            return tmp;
        }
    }
    else { // tensor->dtype == DType::SIZE_T
        if (tensor->mtype == MType::HOST) {
            return ((size_t*)tensor->data)[ind];
        }
        else { // tensor->mtype == MType::CUDA_DEVICE
            float tmp;
            CUDA_CHECK(cudaMemcpy(&tmp, (size_t*)tensor->data + ind, dtype_sz(tensor->dtype), cudaMemcpyDeviceToHost));
            return tmp;
        }
    }
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