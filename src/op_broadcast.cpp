#include <cuda_runtime_api.h>

#include <cstring>

#include "arena.hpp"
#include "cuda_kernels/helpers.hpp"
#include "cuda_kernels/reduce_add.hpp"
#include "operation_priv.hpp"

TensorHandle broadcast(TensorHandle input, size_t n, ArenaAllocatorHandle arena,
                       ComputationContextHandle ctx) {
  size_t *result_dims = (size_t *)arena_alloc(
      arena, sizeof(size_t) * (input->num_dims + 1), alignof(size_t));
  result_dims[0] = n;
  memcpy(result_dims + 1, input->dims, sizeof(size_t) * input->num_dims);
  TensorHandle result =
      tensor_zeroes(input->mtype, input->dtype, input->num_dims + 1,
                    result_dims, false, arena);

  size_t input_size = tensor_size(input);

  if (result->mtype == MType::HOST) {
    for (size_t i = 0; i < n; ++i) {
      if (result->dtype == DType::FLOAT32) {
        memcpy((float *)result->data + (i * input_size), input->data,
               input_size * sizeof(float));
      } else { // result->dtype == DType::SIZE_T
        memcpy((size_t *)result->data + (i * input_size), input->data,
               input_size * sizeof(size_t));
      }
    }
  } else { // result->mtype == MType::CUDA_DEVICE
    for (size_t i = 0; i < n; ++i) {
      if (result->dtype == DType::FLOAT32) {
        CUDA_CHECK(cudaMemcpy((float *)result->data + (i * input_size),
                              input->data, input_size * sizeof(float),
                              cudaMemcpyDeviceToDevice));
      } else { // result->dtype == DType::SIZE_T
        CUDA_CHECK(cudaMemcpy((size_t *)result->data + (i * input_size),
                              input->data, input_size * sizeof(size_t),
                              cudaMemcpyDeviceToDevice));
      }
    }
  }

  update_context_new_op(ctx, {input}, OperationType::BROADCAST, result);

  return result;
}

void broadcast_backward(TensorHandle input, TensorHandle out) {
  size_t n = out->dims[0];
  size_t input_size = tensor_size(input);
  float *in_grads = (float *)input->grads;
  float *out_grads = (float *)out->grads;

  if (input->mtype == MType::HOST) {
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < input_size; ++j) {
        in_grads[j] += out_grads[i * input_size + j];
      }
    }
  } else {
    cuda_reduce_add_vertical(out_grads, in_grads, n, input_size, 1.0f);
  }
}
