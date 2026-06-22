#include <cmath>
#include <cstring>

#include "arena.hpp"
#include "cuda_kernels/logsoftmax.hpp"
#include "operation_priv.hpp"

TensorHandle logsoftmax(TensorHandle left, TensorHandle right,
                        ArenaAllocatorHandle arena,
                        ComputationContextHandle ctx) {
  size_t *result_dims = (size_t *)arena_alloc(
      arena, sizeof(size_t) * (left->num_dims - 1), alignof(size_t));
  memcpy(result_dims, left->dims, sizeof(size_t) * (left->num_dims - 1));
  TensorHandle result = tensor_zeroes(
      left->mtype, left->dtype, left->num_dims - 1, result_dims, false, arena);

  size_t *row_exp_sum_dims = (size_t *)arena_alloc(
      arena, sizeof(size_t) * (left->num_dims - 1), alignof(size_t));
  memcpy(row_exp_sum_dims, left->dims, sizeof(size_t) * (left->num_dims - 1));
  TensorHandle row_exp_sum =
      tensor_zeroes(left->mtype, left->dtype, left->num_dims - 1,
                    row_exp_sum_dims, false, arena);

  size_t *row_max_dims = (size_t *)arena_alloc(
      arena, sizeof(size_t) * (left->num_dims - 1), alignof(size_t));
  memcpy(row_max_dims, left->dims, sizeof(size_t) * (left->num_dims - 1));
  TensorHandle row_max = tensor_zeroes(
      left->mtype, left->dtype, left->num_dims - 1, row_max_dims, false, arena);

  size_t num_sub_tensors = tensor_size(result);
  size_t n = left->dims[left->num_dims - 1];
  float *right_data = (float *)right->data;
  float *left_data = (float *)left->data;
  float *result_data = (float *)result->data;
  float *row_exp_sum_data = (float *)row_exp_sum->data;
  float *row_max_data = (float *)row_max->data;

  if (result->mtype == MType::HOST) {
    for (size_t sub_tensor_ind = 0; sub_tensor_ind < num_sub_tensors;
         ++sub_tensor_ind) {
      float denom = 0;
      size_t target_ind = 0;
      float max_exp = left_data[sub_tensor_ind * n];
      for (size_t i = 1; i < n; ++i) {
        max_exp = std::max(max_exp, left_data[sub_tensor_ind * n + i]);
      }
      for (size_t i = 0; i < n; ++i) {
        denom += std::exp(left_data[sub_tensor_ind * n + i] - max_exp);
        if (right_data[sub_tensor_ind * n + i] == 1.0) {
          target_ind = i;
        }
      }

      result_data[sub_tensor_ind] =
          -left_data[sub_tensor_ind * n + target_ind] + max_exp +
          std::log(denom);
    }
  } else {
    cuda_logsoftmax(left_data, right_data, result_data, row_exp_sum_data,
                    row_max_data, num_sub_tensors, n);
  }

  OperationMetadata op_metadata = {{"row_exp_sum", row_exp_sum},
                                   {"row_max", row_max}};

  update_context_new_op(ctx, {left, right}, OperationType::LOGSOFTMAX, result,
                        op_metadata);

  return result;
}

void logsoftmax_backward(TensorHandle left, TensorHandle right,
                         TensorHandle out, TensorHandle row_exp_sum,
                         TensorHandle row_max) {
  size_t num_sub_tensors = tensor_size(out);
  size_t n = left->dims[left->num_dims - 1];
  float *right_data = (float *)right->data;
  float *left_grads = (float *)left->grads;
  float *left_data = (float *)left->data;
  float *out_grads = (float *)out->grads;
  float *row_exp_sum_data = (float *)row_exp_sum->data;
  float *row_max_data = (float *)row_max->data;

  if (left->mtype == MType::HOST) {
    for (size_t sub_tensor_ind = 0; sub_tensor_ind < num_sub_tensors;
         ++sub_tensor_ind) {
      float denom = 0;
      float max_exp = left_data[sub_tensor_ind * n];
      for (size_t i = 1; i < n; ++i) {
        max_exp = std::max(max_exp, left_data[sub_tensor_ind * n + i]);
      }
      for (size_t i = 0; i < n; ++i) {
        denom += std::exp(left_data[sub_tensor_ind * n + i] - max_exp);
      }
      for (size_t i = 0; i < n; ++i) {
        if (right_data[sub_tensor_ind * n + i] == 1.0) {
          left_grads[sub_tensor_ind * n + i] +=
              (out_grads[sub_tensor_ind] *
               (std::exp(left_data[sub_tensor_ind * n + i] - max_exp) / denom -
                1.0));
        } else {
          left_grads[sub_tensor_ind * n + i] +=
              (out_grads[sub_tensor_ind] *
               (std::exp(left_data[sub_tensor_ind * n + i] - max_exp) / denom));
        }
      }
    }
  } else {
    cuda_logsoftmax_backward(left_data, left_grads, right_data, out_grads,
                             row_exp_sum_data, row_max_data, num_sub_tensors,
                             n);
  }
}
