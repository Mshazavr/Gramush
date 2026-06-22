#include "arena.hpp"
#include "operation_priv.hpp"

TensorHandle embedding(TensorHandle indices, TensorHandle vectors,
                       ArenaAllocatorHandle arena,
                       ComputationContextHandle ctx) {
  return nullptr;
}

void embedding_backward(TensorHandle indices, TensorHandle vectors,
                        TensorHandle out) {}
