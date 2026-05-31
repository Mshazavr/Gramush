#include "operation.hpp"
#include "tensor_priv.hpp"
#include <unordered_map>
#include <vector>
#include <queue>
#include <cmath>
#include <iostream>
#include <optional>
#include <cstring>
#include "arena.hpp"

struct ComputationContext {
    std::vector<OperationType> operation_nodes;

    std::unordered_map<TensorHandle, std::optional<size_t>> tensor_in_edges;
    std::unordered_map<TensorHandle, size_t> tensor_out_count; 
    std::vector<std::vector<TensorHandle>> operation_in_edges;
};


ComputationContextHandle init_computation_context() {
    ComputationContextHandle result = new ComputationContext();
    return result;
}
void computation_context_clear(ComputationContextHandle ctx) {
    ctx->operation_nodes.clear();
    ctx->tensor_in_edges.clear();
    ctx->tensor_out_count.clear();
    ctx->operation_in_edges.clear();
}
void computation_context_free(ComputationContextHandle ctx) {
    delete ctx;
}

void update_context_new_op(
    ComputationContextHandle ctx, 
    std::initializer_list<TensorHandle> inputs, 
    OperationType opnode_t, 
    TensorHandle output
) {
    size_t opnode_h = ctx->operation_nodes.size();
    ctx->operation_nodes.push_back(opnode_t);
    ctx->tensor_in_edges[output] = opnode_h;
    ctx->operation_in_edges.push_back(inputs);
    for (auto tensor_h: inputs) {
        ++ctx->tensor_out_count[tensor_h];
    }
}

TensorHandle mmul(TensorHandle left, TensorHandle right, ArenaAllocatorHandle arena, ComputationContextHandle ctx) {
    // TODO: add dimension validation
    size_t *result_dims = (size_t*)arena_alloc(arena, sizeof(size_t) * left->num_dims, alignof(size_t));
    memcpy(result_dims, left->dims, sizeof(size_t) * (left->num_dims - 2));
    size_t num_dims = left->num_dims;
    result_dims[num_dims - 2] = left->dims[num_dims - 2];
    result_dims[num_dims - 1] = right->dims[num_dims - 1];
    TensorHandle result = tensor_zeroes(left->dtype, num_dims, result_dims, false, arena);

    size_t num_sub_tensors = tensor_num_sub_tensors(left, 2);
    size_t n = left->dims[left->num_dims - 2];
    size_t m = left->dims[left->num_dims - 1];
    size_t k = right->dims[right->num_dims - 1];

    float *left_data = (float*)left->data;
    float *right_data = (float*)right->data;
    float *result_data = (float*)result->data;
    for (size_t sub_tensor_ind = 0; sub_tensor_ind < num_sub_tensors; ++sub_tensor_ind) {
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < k; ++j) {
                for (size_t l = 0; l < m; ++l) {
                    result_data[sub_tensor_ind * n * k + i * k + j] += (
                        left_data[sub_tensor_ind * n * m + i * m + l] * 
                        right_data[sub_tensor_ind * m * k + l * k + j]
                    );
                }
            }
        }
    }
    
    update_context_new_op(ctx, {left, right}, OperationType::MMUL, result);

    return result;
}
void mmul_backward(TensorHandle left, TensorHandle right, TensorHandle out) {
    size_t num_sub_tensors = tensor_num_sub_tensors(left, 2);
    size_t n = left->dims[left->num_dims - 2];
    size_t m = left->dims[left->num_dims - 1];
    size_t k = right->dims[right->num_dims - 1];

    float *left_data = (float*)left->data;
    float *right_data = (float*)right->data;
    float *left_grads = (float*)left->grads;
    float *right_grads = (float*)right->grads;
    float *out_grads = (float*)out->grads;
    for (size_t sub_tensor_ind = 0; sub_tensor_ind < num_sub_tensors; ++sub_tensor_ind) {
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < k; ++j) {
                for (size_t l = 0; l < m; ++l) {
                    // G_left += G_out * R^T
                    left_grads[sub_tensor_ind * n * m + i * m + l] += (
                        out_grads[sub_tensor_ind * n * k + i * k + j] *
                        right_data[sub_tensor_ind * m * k + l * k + j]
                    );

                    // G_right += L^T * G_out
                    right_grads[sub_tensor_ind * m * k + l * k + j] += (
                        out_grads[sub_tensor_ind * n * k + i * k + j] * 
                        left_data[sub_tensor_ind * n * m + i * m + l]
                    );
                }
            }
        }
    }
}

TensorHandle relu(TensorHandle input, ArenaAllocatorHandle arena, ComputationContextHandle ctx) {
    size_t *result_dims = (size_t*)arena_alloc(arena, sizeof(size_t) * input->num_dims, alignof(size_t));
    memcpy(result_dims, input->dims, sizeof(size_t) * input->num_dims);
    TensorHandle result = tensor_zeroes(input->dtype, input->num_dims, result_dims, false, arena);
    size_t n = tensor_size(result);

    float *input_data = (float*)input->data;
    float *result_data = (float*)result->data;
    for (size_t i = 0; i < n; ++i) {
        result_data[i] = std::max(0.0f, input_data[i]);
    }

    update_context_new_op(ctx, {input}, OperationType::RELU, result);

    return result;
}
void relu_backward(TensorHandle input, TensorHandle out) {
    size_t n = tensor_size(out);

    float *input_data = (float*)input->data;
    float *input_grads = (float*)input->grads;
    float *out_grads = (float*)out->grads;
    for (size_t i = 0; i < n; ++i) {
        if (input_data[i] >= 0) {
            input_grads[i] += out_grads[i];
        }
    }
}

TensorHandle softmax(TensorHandle input, ArenaAllocatorHandle arena, ComputationContextHandle ctx) {
    size_t *result_dims = (size_t*)arena_alloc(arena, sizeof(size_t) * input->num_dims, alignof(size_t));
    memcpy(result_dims, input->dims, sizeof(size_t) * input->num_dims);
    TensorHandle result = tensor_zeroes(input->dtype, input->num_dims, result_dims, false, arena);
    size_t num_sub_tensors = tensor_num_sub_tensors(input, 1);
    size_t n = result->dims[result->num_dims - 1];

    float *input_data = (float*)input->data;
    float *result_data = (float*)result->data;
    for (size_t sub_tensor_ind = 0; sub_tensor_ind < num_sub_tensors; ++sub_tensor_ind) {
        float denominator = 0;
        for (size_t i = 0; i < n; ++i) {
            result_data[sub_tensor_ind * n + i] = std::exp(input_data[sub_tensor_ind * n + i]);
            denominator += std::exp(input_data[sub_tensor_ind * n + i]);
        }
        for (size_t i = 0; i < n; ++i) {
            result_data[sub_tensor_ind * n + i] /= denominator;
        }
    }

    update_context_new_op(ctx, {input}, OperationType::SOFTMAX, result);

    return result;
}
void softmax_backward(TensorHandle input, TensorHandle out) {
    size_t num_sub_tensors = tensor_num_sub_tensors(input, 1);
    size_t n = out->dims[out->num_dims - 1];

    float *input_grads = (float*)input->grads;
    float *out_data = (float*)out->data;
    float *out_grads = (float*)out->grads;
    for (size_t sub_tensor_ind = 0; sub_tensor_ind < num_sub_tensors; ++sub_tensor_ind) {
        for (size_t i = 0; i < n; ++i) {
            input_grads[sub_tensor_ind * n + i] += (
                out_data[sub_tensor_ind * n + i] * (1.0 - out_data[sub_tensor_ind * n + i]) * 
                out_grads[sub_tensor_ind * n + i]
            );

            for (size_t j = 0; j < n; ++j) {
                if (j == i) {
                    continue;
                }
                input_grads[sub_tensor_ind * n + i] += (
                    -(out_data[sub_tensor_ind * n + i] * out_data[sub_tensor_ind * n + j]) * 
                    out_grads[sub_tensor_ind * n + j]
                );
            }
        }
    }
}

TensorHandle cross_entropy(TensorHandle left, TensorHandle right, ArenaAllocatorHandle arena, ComputationContextHandle ctx) {
    size_t *result_dims = (size_t*)arena_alloc(arena, sizeof(size_t) * (left->num_dims - 1), alignof(size_t));
    memcpy(result_dims, left->dims, sizeof(size_t) * (left->num_dims - 1));
    TensorHandle result = tensor_zeroes(left->dtype, left->num_dims - 1, result_dims, false, arena);

    size_t num_sub_tensors = tensor_size(result);
    size_t n = left->dims[left->num_dims - 1];
    float *right_data = (float*)right->data;
    float *left_data = (float*)left->data;
    float *result_data = (float*)result->data;
    for (size_t sub_tensor_ind = 0; sub_tensor_ind < num_sub_tensors; ++sub_tensor_ind) {
        for (size_t i = 0; i < n; ++i) {
            if (right_data[sub_tensor_ind * n + i] == 1.0) {
                result_data[sub_tensor_ind] = -std::log(left_data[sub_tensor_ind * n + i]);
            }
        }
    }

    update_context_new_op(ctx, {left, right}, OperationType::CROSS_ENTROPY, result);

    return result;
}
void cross_entropy_backward(TensorHandle left, TensorHandle right, TensorHandle out) {
    size_t num_sub_tensors = tensor_size(out);
    size_t n = left->dims[left->num_dims - 1];
    float *right_data = (float*)right->data;
    float *left_grads = (float*)left->grads;
    float *left_data = (float*)left->data;
    float *out_grads = (float*)out->grads;
    for (size_t sub_tensor_ind = 0; sub_tensor_ind < num_sub_tensors; ++sub_tensor_ind) {
        for (size_t i = 0; i < n; ++i) {
            if (right_data[sub_tensor_ind * n + i] == 1.0) {
                left_grads[sub_tensor_ind * n + i] += (
                    -out_grads[sub_tensor_ind] / left_data[sub_tensor_ind * n + i]
                );
            }
        }
    }
}

TensorHandle mean(TensorHandle input, ArenaAllocatorHandle arena, ComputationContextHandle ctx) {
    size_t *result_dims = (size_t*)arena_alloc(arena, sizeof(size_t) * (input->num_dims - 1), alignof(size_t));
    memcpy(result_dims, input->dims, sizeof(size_t) * (input->num_dims - 1));
    TensorHandle result = tensor_zeroes(input->dtype, input->num_dims - 1, result_dims, false, arena);

    size_t num_sub_tensors = tensor_size(result);
    size_t n = input->dims[input->num_dims - 1];
    float *input_data = (float*)input->data;
    float *result_data = (float*)result->data;
    for (size_t sub_tensor_ind = 0; sub_tensor_ind < num_sub_tensors; ++sub_tensor_ind) {
        for (size_t i = 0; i < n; ++i) {
            result_data[sub_tensor_ind] += input_data[sub_tensor_ind * n + i];
        }
        result_data[sub_tensor_ind] /= (float)n;
    }

    update_context_new_op(ctx, {input}, OperationType::MEAN, result);

    return result;
}
void mean_backward(TensorHandle input, TensorHandle out) {
    size_t num_sub_tensors = tensor_size(out);
    size_t n = input->dims[input->num_dims - 1];
    float *input_grads = (float*)input->grads;
    float *out_grads = (float*)out->grads;
    for (size_t sub_tensor_ind = 0; sub_tensor_ind < num_sub_tensors; ++sub_tensor_ind) {
        for (size_t i = 0; i < n; ++i) {
            input_grads[sub_tensor_ind * n + i] += out_grads[sub_tensor_ind] / (float)n;
        }
    }
}


TensorHandle addition(TensorHandle left, TensorHandle right, ArenaAllocatorHandle arena, ComputationContextHandle ctx) {
    size_t *result_dims = (size_t*)arena_alloc(arena, sizeof(size_t) * left->num_dims, alignof(size_t));
    memcpy(result_dims, left->dims, sizeof(size_t) * left->num_dims);
    TensorHandle result = tensor_zeroes(left->dtype, left->num_dims, result_dims, false, arena);
    size_t n = tensor_size(result);
    float *left_data = (float*)left->data;
    float *right_data = (float*)right->data;
    float *result_data = (float*)result->data;
    for (size_t i = 0; i < n; ++i) {
        result_data[i] = left_data[i] + right_data[i];
    }

    update_context_new_op(ctx, {left, right}, OperationType::ADDITION, result);

    return result;
}
void addition_backward(TensorHandle left, TensorHandle right, TensorHandle out) {
    size_t n = tensor_size(out);
    float *left_grads = (float*)left->grads;
    float *right_grads = (float*)right->grads;
    float *out_grads = (float*)out->grads;
    for (size_t i = 0; i < n; ++i) {
        left_grads[i] += out_grads[i];
        right_grads[i] += out_grads[i];
    }
}

TensorHandle broadcast(TensorHandle input, size_t n, ArenaAllocatorHandle arena, ComputationContextHandle ctx) {
    size_t *result_dims = (size_t*)arena_alloc(arena, sizeof(size_t) * (input->num_dims + 1), alignof(size_t));
    result_dims[0] = n;
    memcpy(result_dims + 1, input->dims, sizeof(size_t) * input->num_dims);
    TensorHandle result = tensor_zeroes(input->dtype, input->num_dims + 1, result_dims, false, arena);

    size_t input_size = tensor_size(input);
    for (size_t i = 0; i < n; ++i) {
        if (result->dtype == DType::FLOAT32) {
            memcpy((float*)result->data + (i * input_size), input->data, input_size * sizeof(float));
        } else { // result->dtype == DType::SIZE_T
            memcpy((size_t*)result->data + (i * input_size), input->data, input_size * sizeof(size_t));
        }
    }

    update_context_new_op(ctx, {input}, OperationType::BROADCAST, result);

    return result;
}
void broadcast_backward(TensorHandle input, TensorHandle out) {
    size_t n = out->dims[0];
    size_t input_size = tensor_size(input);
    float *in_grads = (float*)input->grads;
    float *out_grads = (float*)out->grads;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < input_size; ++j) {
            in_grads[j] += out_grads[i * input_size + j];
        }
    }
}


void backward_single(TensorHandle tensor, ComputationContextHandle ctx) {
    size_t opnode_h = ctx->tensor_in_edges[tensor].value();
    if (ctx->operation_nodes[opnode_h] == OperationType::MMUL) {
        mmul_backward(
            ctx->operation_in_edges[opnode_h][0], 
            ctx->operation_in_edges[opnode_h][1], 
            tensor
        );
    }
    else if (ctx->operation_nodes[opnode_h] == OperationType::RELU) {
        relu_backward(ctx->operation_in_edges[opnode_h][0], tensor);
    }
    else if (ctx->operation_nodes[opnode_h] == OperationType::SOFTMAX) {
        softmax_backward(ctx->operation_in_edges[opnode_h][0], tensor);
    }
    else if (ctx->operation_nodes[opnode_h] == OperationType::CROSS_ENTROPY) {
        cross_entropy_backward(
            ctx->operation_in_edges[opnode_h][0], 
            ctx->operation_in_edges[opnode_h][1], 
            tensor
        );
    }
    else if (ctx->operation_nodes[opnode_h] == OperationType::MEAN) {
        mean_backward(ctx->operation_in_edges[opnode_h][0], tensor);
    }
    else if (ctx->operation_nodes[opnode_h] == OperationType::ADDITION) {
        addition_backward(
            ctx->operation_in_edges[opnode_h][0], 
            ctx->operation_in_edges[opnode_h][1], 
            tensor
        );
    }
    else if (ctx->operation_nodes[opnode_h] == OperationType::BROADCAST) {
        broadcast_backward(ctx->operation_in_edges[opnode_h][0], tensor);
    }
    else {
        // TODO clean this up
        exit(0);
    }
}

void backward(TensorHandle tensor, ComputationContextHandle ctx) {
    // TODO: assert dims = []
    ((float*)(tensor->grads))[0] = 1.0;
    
    std::queue<TensorHandle> tensor_queue; 
    tensor_queue.push(tensor);
    while (!tensor_queue.empty()) {
        TensorHandle current_tensor = tensor_queue.front();
        tensor_queue.pop();

        std::optional<size_t> opnode_h_op = ctx->tensor_in_edges[current_tensor];
        if (!opnode_h_op) {
            continue;
        }

        backward_single(current_tensor, ctx);
        size_t opnode_h = opnode_h_op.value();

        for (const auto &in_tensor: ctx->operation_in_edges[opnode_h]) {
            --ctx->tensor_out_count[in_tensor];
            if (ctx->tensor_out_count[in_tensor] == 0) {
                tensor_queue.push(in_tensor);
            }
        }
    }
}

void optimize(std::vector<TensorHandle> &tensors, float beta) {
    for(auto tensor: tensors) {
        size_t n = tensor_size(tensor);
        float *data = (float*)tensor->data;
        float *grads = (float*)tensor->grads;
        for (size_t i = 0; i < n; ++i) {
            data[i] -= grads[i] * beta;
            grads[i] = 0;
        }
    } 
}