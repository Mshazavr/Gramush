#pragma once 

#include <vector>
#include "tensor.hpp"

enum class OperationType {
    MMUL,
    ADDITION,
    SUBTRACTION, 
    RELU,
    SOFTMAX,
    CROSS_ENTROPY,
    LOGSOFTMAX,
    MEAN,
    BROADCAST,
    EMBEDDING
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
TensorHandle logsoftmax(TensorHandle left, TensorHandle right, ArenaAllocatorHandle arena, ComputationContextHandle ctx); 
TensorHandle mean(TensorHandle input, ArenaAllocatorHandle arena, ComputationContextHandle ctx); 
TensorHandle addition(TensorHandle left, TensorHandle right, ArenaAllocatorHandle arena, ComputationContextHandle ctx); 
TensorHandle broadcast(TensorHandle input, size_t n, ArenaAllocatorHandle arena, ComputationContextHandle ctx); 
TensorHandle embedding(TensorHandle indices, TensorHandle vectors, ArenaAllocatorHandle arena, ComputationContextHandle ctx); 

void backward(TensorHandle final, ComputationContextHandle ctx);
void optimize(std::vector<TensorHandle> &tensors, float beta);
