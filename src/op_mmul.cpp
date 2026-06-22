#include <cstring>

#include "arena.hpp"
#include "cuda_kernels/batch_mmul.hpp"
#include "operation_priv.hpp"

TensorHandle mmul(TensorHandle left, TensorHandle right,
                  ArenaAllocatorHandle arena, ComputationContextHandle ctx) {
  // TODO: add dimension validation
  size_t *result_dims = (size_t *)arena_alloc(
      arena, sizeof(size_t) * left->num_dims, alignof(size_t));
  memcpy(result_dims, left->dims, sizeof(size_t) * (left->num_dims - 2));
  size_t num_dims = left->num_dims;
  result_dims[num_dims - 2] = left->dims[num_dims - 2];
  result_dims[num_dims - 1] = right->dims[num_dims - 1];
  TensorHandle result = tensor_zeroes(left->mtype, left->dtype, num_dims,
                                      result_dims, false, arena);

  size_t num_sub_tensors = tensor_num_sub_tensors(left, 2);
  size_t n = left->dims[left->num_dims - 2];
  size_t m = left->dims[left->num_dims - 1];
  size_t k = right->dims[right->num_dims - 1];

  float *left_data = (float *)left->data;
  float *right_data = (float *)right->data;
  float *result_data = (float *)result->data;

  if (result->mtype == MType::HOST) {
    for (size_t sub_tensor_ind = 0; sub_tensor_ind < num_sub_tensors;
         ++sub_tensor_ind) {
      for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < k; ++j) {
          for (size_t l = 0; l < m; ++l) {
            result_data[sub_tensor_ind * n * k + i * k + j] +=
                (left_data[sub_tensor_ind * n * m + i * m + l] *
                 right_data[sub_tensor_ind * m * k + l * k + j]);
          }
        }
      }
    }
  } else {
    cuda_batch_mmul(left_data, false, right_data, false, result_data,
                    num_sub_tensors, n, m, k, 1.0, 0.0);
  }

  update_context_new_op(ctx, {left, right}, OperationType::MMUL, result);

  return result;
}

void mmul_backward(TensorHandle left, TensorHandle right, TensorHandle out) {
  size_t num_sub_tensors = tensor_num_sub_tensors(left, 2);
  size_t n = left->dims[left->num_dims - 2];
  size_t m = left->dims[left->num_dims - 1];
  size_t k = right->dims[right->num_dims - 1];

  float *left_data = (float *)left->data;
  float *right_data = (float *)right->data;
  float *left_grads = (float *)left->grads;
  float *right_grads = (float *)right->grads;
  float *out_grads = (float *)out->grads;

  if (left->mtype == MType::HOST) {
    for (size_t sub_tensor_ind = 0; sub_tensor_ind < num_sub_tensors;
         ++sub_tensor_ind) {
      for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < k; ++j) {
          for (size_t l = 0; l < m; ++l) {
            // G_left += G_out * R^T
            left_grads[sub_tensor_ind * n * m + i * m + l] +=
                (out_grads[sub_tensor_ind * n * k + i * k + j] *
                 right_data[sub_tensor_ind * m * k + l * k + j]);

            // G_right += L^T * G_out
            right_grads[sub_tensor_ind * m * k + l * k + j] +=
                (out_grads[sub_tensor_ind * n * k + i * k + j] *
                 left_data[sub_tensor_ind * n * m + i * m + l]);
          }
        }
      }
    }
  } else {
    cuda_batch_mmul(out_grads, false, right_data, true, left_grads,
                    num_sub_tensors, n, k, m, 1.0, 1.0);
    cuda_batch_mmul(left_data, true, out_grads, false, right_grads,
                    num_sub_tensors, m, n, k, 1.0, 1.0);
  }
}
