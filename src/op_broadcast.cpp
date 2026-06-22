#include <cuda_runtime_api.h>

#include <cstdlib>
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

  switch (result->mtype) {
  case MType::HOST:
    for (size_t i = 0; i < n; ++i) {
      switch (result->dtype) {
      case DType::FLOAT32:
        memcpy((float *)result->data + (i * input_size), input->data,
               input_size * sizeof(float));
        break;
      case DType::SIZE_T:
        memcpy((size_t *)result->data + (i * input_size), input->data,
               input_size * sizeof(size_t));
        break;
      default:
        __builtin_unreachable();
        exit(0);
      }
    }
    break;
  case MType::CUDA_DEVICE:
    for (size_t i = 0; i < n; ++i) {
      switch (result->dtype) {
      case DType::FLOAT32:
        CUDA_CHECK(cudaMemcpy((float *)result->data + (i * input_size),
                              input->data, input_size * sizeof(float),
                              cudaMemcpyDeviceToDevice));
        break;
      case DType::SIZE_T:
        CUDA_CHECK(cudaMemcpy((size_t *)result->data + (i * input_size),
                              input->data, input_size * sizeof(size_t),
                              cudaMemcpyDeviceToDevice));
        break;
      default:
        __builtin_unreachable();
        exit(0);
      }
    }
    break;
  default:
    __builtin_unreachable();
    exit(0);
  }

  update_context_new_op(ctx, {input}, OperationType::BROADCAST, result);

  return result;
}

void broadcast_backward(TensorHandle input, TensorHandle out) {
  size_t n = out->dims[0];
  size_t input_size = tensor_size(input);
  float *in_grads = (float *)input->grads;
  float *out_grads = (float *)out->grads;

  switch (input->mtype) {
  case MType::HOST:
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < input_size; ++j) {
        in_grads[j] += out_grads[i * input_size + j];
      }
    }
    break;
  case MType::CUDA_DEVICE:
    cuda_reduce_add_vertical(out_grads, in_grads, n, input_size, 1.0f);
    break;
  default:
    __builtin_unreachable();
    exit(0);
  }
}
