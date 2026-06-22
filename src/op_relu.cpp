#include <algorithm>
#include <cstring>

#include "arena.hpp"
#include "cuda_kernels/relu.hpp"
#include "operation_priv.hpp"

TensorHandle relu(TensorHandle input, ArenaAllocatorHandle arena,
                  ComputationContextHandle ctx) {
  size_t *result_dims = (size_t *)arena_alloc(
      arena, sizeof(size_t) * input->num_dims, alignof(size_t));
  memcpy(result_dims, input->dims, sizeof(size_t) * input->num_dims);
  TensorHandle result = tensor_zeroes(
      input->mtype, input->dtype, input->num_dims, result_dims, false, arena);
  size_t n = tensor_size(result);

  float *input_data = (float *)input->data;
  float *result_data = (float *)result->data;

  if (result->mtype == MType::HOST) {
    for (size_t i = 0; i < n; ++i) {
      result_data[i] = std::max(0.0f, input_data[i]);
    }
  } else {
    cuda_relu(input_data, result_data, n);
  }

  update_context_new_op(ctx, {input}, OperationType::RELU, result);

  return result;
}

void relu_backward(TensorHandle input, TensorHandle out) {
  size_t n = tensor_size(out);

  float *input_data = (float *)input->data;
  float *input_grads = (float *)input->grads;
  float *out_grads = (float *)out->grads;

  if (input->mtype == MType::HOST) {
    for (size_t i = 0; i < n; ++i) {
      if (input_data[i] >= 0) {
        input_grads[i] += out_grads[i];
      }
    }
  } else {
    cuda_relu_backward(input_data, input_grads, out_grads, n);
  }
}
