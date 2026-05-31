#pragma once 

#include <vector>
#include "tensor.hpp"

enum class OperationType {
    MMUL,

    // element-wise
    ADDITION,
    SUBTRACTION, 
    RELU,

    // across the last dimension
    SOFTMAX,

    // reduction along the last dimension
    CROSS_ENTROPY,
    MEAN,

    BROADCAST
};

struct ComputationContext;

using ComputationContextHandle = ComputationContext*;

ComputationContextHandle init_computation_context();
void computation_context_clear(ComputationContextHandle ctx);
void computation_context_free(ComputationContextHandle ctx);

TensorHandle mmul(TensorHandle left, TensorHandle right, ArenaAllocatorHandle arena, ComputationContextHandle ctx); 
TensorHandle relu(TensorHandle input, ArenaAllocatorHandle arena, ComputationContextHandle ctx); 
TensorHandle softmax(TensorHandle input, ArenaAllocatorHandle arena, ComputationContextHandle ctx); 
TensorHandle cross_entropy(TensorHandle left, TensorHandle right, ArenaAllocatorHandle arena, ComputationContextHandle ctx); 
TensorHandle mean(TensorHandle input, ArenaAllocatorHandle arena, ComputationContextHandle ctx); 
TensorHandle addition(TensorHandle left, TensorHandle right, ArenaAllocatorHandle arena, ComputationContextHandle ctx); 
TensorHandle broadcast(TensorHandle input, size_t n, ArenaAllocatorHandle arena, ComputationContextHandle ctx); 

void backward(TensorHandle final, ComputationContextHandle ctx);
void optimize(std::vector<TensorHandle> &tensors, float beta);
