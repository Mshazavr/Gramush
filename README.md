# Gramush autograd library

Gramush is an educational, experimental automatic differentiation library written in `C++`. It aims to provide a PyTorch-like, intuitive tensor API while keeping control flow explicit so AI engineers can work productively in C++ and low-level optimizers can see exactly what runs where.

## Installation

```bash
git clone ...
cd Gramush
make
```

This builds the `mshon` binary in the project root. Run `make clean` to remove build artifacts.

### Requirements

- **Compiler:** `g++` with C++23 support
- **Build:** GNU Make

To use GPU-backed tensors and CUDA kernels, you also need:

- **CUDA Toolkit** (provides `nvcc` and `libcudart`)
- An **NVIDIA GPU** with a driver compatible with your CUDA version

By default the Makefile expects CUDA at `/usr/local/cuda`. Override paths or architecture as needed:

```bash
make CUDA_HOME=/opt/cuda CUDA_ARCH=sm_89
```

`CUDA_ARCH` should match your GPU (e.g. `sm_86` for RTX 30-series, `sm_89` for RTX 40-series). CPU-only code paths work without a GPU; set `MType::HOST` when creating tensors.

## Examples

### CPU (no CUDA kernels)

Operations run as plain C++ loops on host memory:

```cpp
#include "arena.hpp"
#include "operation.hpp"
#include "tensor.hpp"

int main() {
  ComputationContextHandle ctx = init_computation_context();
  ArenaAllocatorHandle arena = arena_init(1 << 20, 0);

  TensorHandle a = tensor_from_vector(MType::HOST, DType::FLOAT32, {2, 2}, true,
                                      {1.f, 2.f, 3.f, 4.f}, arena);
  TensorHandle b = tensor_from_vector(MType::HOST, DType::FLOAT32, {2, 2}, true,
                                      {0.5f, 0.5f, 0.5f, 0.5f}, arena);

  TensorHandle c = addition(a, b, arena, ctx);
  TensorHandle loss = mean(c, arena, ctx);

  backward(loss, ctx);
  std::vector<TensorHandle> params = {a, b};
  optimize(params, 0.1f);

  computation_context_free(ctx);
  arena_free(arena);
  return 0;
}
```

### GPU (CUDA kernels)

The same API works on device memory — kernels are dispatched transparently via `MType::CUDA_DEVICE`:

```cpp
#include "arena.hpp"
#include "operation.hpp"
#include "tensor.hpp"

int main() {
  ComputationContextHandle ctx = init_computation_context();
  ArenaAllocatorHandle arena = arena_init(1 << 20, 1 << 20);  // host + device bytes

  TensorHandle x = tensor_random(MType::CUDA_DEVICE, DType::FLOAT32, {32, 784}, false, arena);
  TensorHandle w = tensor_random(MType::CUDA_DEVICE, DType::FLOAT32, {784, 128}, true, arena);
  TensorHandle b = tensor_random(MType::CUDA_DEVICE, DType::FLOAT32, {128}, true, arena);

  TensorHandle logits = addition(mmul(x, w, arena, ctx),
                                 broadcast(b, 32, arena, ctx), arena, ctx);
  TensorHandle out = relu(logits, arena, ctx);
  TensorHandle loss = mean(out, arena, ctx);

  backward(loss, ctx);
  std::vector<TensorHandle> params = {w, b};
  optimize(params, 0.01f);

  computation_context_free(ctx);
  arena_free(arena);
  return 0;
}
```

See `src/main.cpp` for a full MNIST training loop on CUDA.
