#pragma once

#include "tensor.hpp"

struct Tensor {
    size_t num_dims;
    size_t *dims;
    size_t *strides;
    float *data;
    float *grads;
    bool learnable; 
};