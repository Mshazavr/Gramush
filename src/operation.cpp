#include <cuda_runtime_api.h>

#include <iostream>
#include <optional>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

#include "arena.hpp"
#include "cuda_kernels/add.hpp"
#include "cuda_kernels/helpers.hpp"
#include "operation_priv.hpp"

#define LESS_CUDA false

struct ComputationContext {
  std::vector<OperationType> operation_nodes;
  std::vector<OperationMetadata> operation_metadata;

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
void computation_context_free(ComputationContextHandle ctx) { delete ctx; }

void update_context_new_op(ComputationContextHandle ctx,
                           std::initializer_list<TensorHandle> inputs,
                           OperationType opnode_t, TensorHandle output,
                           OperationMetadata op_metadata) {
  size_t opnode_h = ctx->operation_nodes.size();
  ctx->operation_nodes.push_back(opnode_t);
  ctx->operation_metadata.push_back(op_metadata);
  ctx->tensor_in_edges[output] = opnode_h;
  ctx->operation_in_edges.push_back(inputs);
  for (auto tensor_h : inputs) {
    ++ctx->tensor_out_count[tensor_h];
  }
}

void backward_single(TensorHandle tensor, ComputationContextHandle ctx) {
  size_t opnode_h = ctx->tensor_in_edges[tensor].value();
  if (ctx->operation_nodes[opnode_h] == OperationType::MMUL) {
    mmul_backward(ctx->operation_in_edges[opnode_h][0],
                  ctx->operation_in_edges[opnode_h][1], tensor);
  } else if (ctx->operation_nodes[opnode_h] == OperationType::RELU) {
    relu_backward(ctx->operation_in_edges[opnode_h][0], tensor);
  } else if (ctx->operation_nodes[opnode_h] == OperationType::SOFTMAX) {
    softmax_backward(ctx->operation_in_edges[opnode_h][0], tensor);
  } else if (ctx->operation_nodes[opnode_h] == OperationType::CROSS_ENTROPY) {
    cross_entropy_backward(ctx->operation_in_edges[opnode_h][0],
                           ctx->operation_in_edges[opnode_h][1], tensor);
  } else if (ctx->operation_nodes[opnode_h] == OperationType::LOGSOFTMAX) {
    logsoftmax_backward(ctx->operation_in_edges[opnode_h][0],
                        ctx->operation_in_edges[opnode_h][1], tensor,
                        ctx->operation_metadata[opnode_h]["row_exp_sum"],
                        ctx->operation_metadata[opnode_h]["row_max"]);
  } else if (ctx->operation_nodes[opnode_h] == OperationType::MEAN) {
    mean_backward(ctx->operation_in_edges[opnode_h][0], tensor);
  } else if (ctx->operation_nodes[opnode_h] == OperationType::ADDITION) {
    addition_backward(ctx->operation_in_edges[opnode_h][0],
                      ctx->operation_in_edges[opnode_h][1], tensor);
  } else if (ctx->operation_nodes[opnode_h] == OperationType::BROADCAST) {
    broadcast_backward(ctx->operation_in_edges[opnode_h][0], tensor);
  } else if (ctx->operation_nodes[opnode_h] == OperationType::EMBEDDING) {
    embedding_backward(ctx->operation_in_edges[opnode_h][0],
                       ctx->operation_in_edges[opnode_h][1], tensor);
  } else {
    // TODO clean this up
    exit(0);
  }
}

void backward(TensorHandle tensor, ComputationContextHandle ctx) {
  // TODO: assert dims = []
  if (tensor->mtype == MType::HOST) {
    ((float *)(tensor->grads))[0] = 1.0;
  } else { // tensor->mtype == MType::CUDA_DEVICE
    float tmp = 1.0f;
    CUDA_CHECK(cudaMemcpy(tensor->grads, &tmp, dtype_sz(tensor->dtype),
                          cudaMemcpyHostToDevice));
  }

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

    for (const auto &in_tensor : ctx->operation_in_edges[opnode_h]) {
      --ctx->tensor_out_count[in_tensor];
      if (ctx->tensor_out_count[in_tensor] == 0) {
        tensor_queue.push(in_tensor);
      }
    }
  }
}

void optimize(std::vector<TensorHandle> &tensors, float beta) {
  for (auto tensor : tensors) {
    size_t n = tensor_size(tensor);
    float *data = (float *)tensor->data;
    float *grads = (float *)tensor->grads;

    if (tensor->mtype == MType::HOST) {
      for (size_t i = 0; i < n; ++i) {
        data[i] -= grads[i] * beta;
        grads[i] = 0;
      }
    } else { // tensor->mtype == MType::CUDA_DEVICE
      cuda_add(data, grads, data, n, 1.0, -beta);
      CUDA_CHECK(cudaMemset(grads, 0.0f, n * dtype_sz(tensor->dtype)));
    }
  }
}
