#include "operation_priv.hpp"
#include "arena.hpp"
#include <cmath>
#include <cstring>

TensorHandle cross_entropy(TensorHandle left, TensorHandle right, ArenaAllocatorHandle arena, ComputationContextHandle ctx) {
    size_t *result_dims = (size_t*)arena_alloc(arena, sizeof(size_t) * (left->num_dims - 1), alignof(size_t));
    memcpy(result_dims, left->dims, sizeof(size_t) * (left->num_dims - 1));
    TensorHandle result = tensor_zeroes(left->mtype, left->dtype, left->num_dims - 1, result_dims, false, arena);

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
