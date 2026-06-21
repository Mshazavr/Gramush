#include "operation_priv.hpp"
#include "arena.hpp"
#include <cstring>
#include "cuda_kernels/add.hpp"

TensorHandle addition(TensorHandle left, TensorHandle right, ArenaAllocatorHandle arena, ComputationContextHandle ctx) {
    size_t *result_dims = (size_t*)arena_alloc(arena, sizeof(size_t) * left->num_dims, alignof(size_t));
    memcpy(result_dims, left->dims, sizeof(size_t) * left->num_dims);
    TensorHandle result = tensor_zeroes(left->mtype, left->dtype, left->num_dims, result_dims, false, arena);
    size_t n = tensor_size(result);
    float *left_data = (float*)left->data;
    float *right_data = (float*)right->data;
    float *result_data = (float*)result->data;
    
    if (result->mtype == MType::HOST) {
        for (size_t i = 0; i < n; ++i) {
            result_data[i] = left_data[i] + right_data[i];
        }
    }
    else {
        cuda_add(left_data, right_data, result_data, n, 1.0, 1.0);
    }

    update_context_new_op(ctx, {left, right}, OperationType::ADDITION, result);

    return result;
}

void addition_backward(TensorHandle left, TensorHandle right, TensorHandle out) {
    size_t n = tensor_size(out);
    float *left_grads = (float*)left->grads;
    float *right_grads = (float*)right->grads;
    float *out_grads = (float*)out->grads;
    
    if (left->mtype == MType::HOST) {
        for (size_t i = 0; i < n; ++i) {
            left_grads[i] += out_grads[i];
            right_grads[i] += out_grads[i];
        }
    }
    else {
        cuda_add(left_grads, out_grads, left_grads, n, 1.0, 1.0);
        cuda_add(right_grads, out_grads, right_grads, n, 1.0, 1.0);
    }
}
