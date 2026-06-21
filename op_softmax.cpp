#include "operation_priv.hpp"
#include "arena.hpp"
#include <cmath>
#include <cstring>

TensorHandle softmax(TensorHandle input, ArenaAllocatorHandle arena, ComputationContextHandle ctx) {
    size_t *result_dims = (size_t*)arena_alloc(arena, sizeof(size_t) * input->num_dims, alignof(size_t));
    memcpy(result_dims, input->dims, sizeof(size_t) * input->num_dims);
    TensorHandle result = tensor_zeroes(input->mtype, input->dtype, input->num_dims, result_dims, false, arena);
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
