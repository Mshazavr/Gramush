#include "tensor_priv.hpp"
#include <vector>
#include <cstdlib>
#include <iostream>

TensorHandle tensor_zeroes(const std::vector<size_t> &dims, bool learnable) {
    size_t num_elems = 1;
    for (const auto &dim: dims) {
        num_elems *= dim;
    }

    float *data = new float[num_elems]();
    float *grads = new float[num_elems]();

    std::vector<size_t> strides; 
    strides.reserve(dims.size());
    for (const auto &dim: dims) {
        num_elems /= dim;
        strides.push_back(num_elems);
    }

    TensorHandle result = new Tensor {
        .dims = dims,
        .strides = strides,
        .data = data,
        .grads = grads, 
        .learnable = learnable
    };
    return result;
}


TensorHandle tensor_random(const std::vector<size_t> &dims, bool learnable) {
    TensorHandle result = tensor_zeroes(dims, learnable);
    size_t n = tensor_size(result);
    for (size_t i = 0; i < n; ++i) {
        result->data[i] = (std::rand() / (float)RAND_MAX) * 0.2f - 0.1f;
    }
    return result;
}

TensorHandle tensor_from_vector(const std::vector<size_t> &dims, bool learnable, const std::vector<float> &data) {
    TensorHandle result = tensor_zeroes(dims, learnable);
    memcpy(result->data, data.data(), data.size() * sizeof(float));
    return result;
}

TensorHandle tensor_from_data(const std::vector<size_t> &dims, bool learnable, const float* data) {
    TensorHandle result = tensor_zeroes(dims, learnable);
    memcpy(result->data, data, tensor_size(result) * sizeof(float));
    return result;   
}

void tensor_free(TensorHandle tensor) {
    delete[] tensor->data;
    delete[] tensor->grads;
    delete tensor;
}

float tensor_at(TensorHandle tensor, const std::vector<size_t> &indices) {
    size_t ind = 0;
    for (size_t i = 0; i < tensor->strides.size(); ++i) {
        ind += tensor->strides[i] * indices[i];
    }
    return tensor->data[ind];
}

size_t tensor_num_sub_tensors(TensorHandle tensor, size_t num_last_dims) {
    size_t num_sub_tensors = 1;
    for (size_t i = 0; i + num_last_dims < tensor->dims.size(); ++i) {
        num_sub_tensors *= tensor->dims[i];
    }
    return num_sub_tensors;
}

size_t tensor_size(TensorHandle tensor) {
    return tensor_num_sub_tensors(tensor, 0);
}