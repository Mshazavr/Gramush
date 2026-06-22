#pragma once

#include <initializer_list>

#include "operation.hpp"
#include "tensor_priv.hpp"

void update_context_new_op(ComputationContextHandle ctx,
                           std::initializer_list<TensorHandle> inputs,
                           OperationType opnode_t, TensorHandle output,
                           OperationMetadata op_metadata = OperationMetadata());

void mmul_backward(TensorHandle left, TensorHandle right, TensorHandle out);
void relu_backward(TensorHandle input, TensorHandle out);
void softmax_backward(TensorHandle input, TensorHandle out);
void cross_entropy_backward(TensorHandle left, TensorHandle right,
                            TensorHandle out);
void logsoftmax_backward(TensorHandle left, TensorHandle right,
                         TensorHandle out, TensorHandle row_exp_sum,
                         TensorHandle row_max);
void mean_backward(TensorHandle input, TensorHandle out);
void addition_backward(TensorHandle left, TensorHandle right, TensorHandle out);
void broadcast_backward(TensorHandle input, TensorHandle out);
void embedding_backward(TensorHandle indices, TensorHandle vectors,
                        TensorHandle out);
