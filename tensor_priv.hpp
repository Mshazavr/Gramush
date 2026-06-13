#pragma once

#include "tensor.hpp"

struct Tensor {
    size_t num_dims;
    size_t *dims;
    size_t *strides;
    void *data;
    void *grads;
    bool learnable; 
    DType dtype;
    MType mtype;
};

size_t dtype_sz(DType dtype);
size_t dtype_al(DType dtype);