#pragma once

#include "tensor.hpp"

struct Tensor {
    const std::vector<size_t> dims; 
    const std::vector<size_t> strides; 
    float *const data;
    float *const grads;
    const bool learnable; 
};