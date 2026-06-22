#include <cstdlib>
#include <cstring>

#include "arena.hpp"
#include "cuda_kernels/reduce_add.hpp"
#include "operation_priv.hpp"

TensorHandle mean(TensorHandle input, ArenaAllocatorHandle arena,
                  ComputationContextHandle ctx) {
  size_t *result_dims = (size_t *)arena_alloc(
      arena, sizeof(size_t) * (input->num_dims - 1), alignof(size_t));
  memcpy(result_dims, input->dims, sizeof(size_t) * (input->num_dims - 1));
  TensorHandle result =
      tensor_zeroes(input->mtype, input->dtype, input->num_dims - 1,
                    result_dims, false, arena);

  size_t num_sub_tensors = tensor_size(result);
  size_t n = input->dims[input->num_dims - 1];
  float *input_data = (float *)input->data;
  float *result_data = (float *)result->data;

  switch (result->mtype) {
  case MType::HOST:
    for (size_t sub_tensor_ind = 0; sub_tensor_ind < num_sub_tensors;
         ++sub_tensor_ind) {
      for (size_t i = 0; i < n; ++i) {
        result_data[sub_tensor_ind] += input_data[sub_tensor_ind * n + i];
      }
      result_data[sub_tensor_ind] /= (float)n;
    }
    break;
  case MType::CUDA_DEVICE:
    cuda_reduce_add(input_data, result_data, num_sub_tensors, n,
                    1.0f / ((float)n));
    break;
  default:
    __builtin_unreachable();
    exit(0);
  }

  update_context_new_op(ctx, {input}, OperationType::MEAN, result);

  return result;
}

void mean_backward(TensorHandle input, TensorHandle out) {
  size_t num_sub_tensors = tensor_size(out);
  size_t n = input->dims[input->num_dims - 1];
  float *input_grads = (float *)input->grads;
  float *out_grads = (float *)out->grads;

  switch (input->mtype) {
  case MType::HOST:
    for (size_t sub_tensor_ind = 0; sub_tensor_ind < num_sub_tensors;
         ++sub_tensor_ind) {
      for (size_t i = 0; i < n; ++i) {
        input_grads[sub_tensor_ind * n + i] +=
            out_grads[sub_tensor_ind] / (float)n;
      }
    }
    break;
  case MType::CUDA_DEVICE:
    cuda_reduce_add_backward(input_grads, out_grads, num_sub_tensors, n,
                             1.0f / ((float)n));
    break;
  default:
    __builtin_unreachable();
    exit(0);
  }
}
