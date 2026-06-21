# Compiler and compilation flags
CXX = g++
SRCDIR = src
BUILDDIR = build
CXXFLAGS = -O3 -Wall -Wextra -std=c++23 -I/usr/local/cuda/include -I$(SRCDIR)
DEBUGFLAGS = -g -fno-omit-frame-pointer -fsanitize=undefined
ALLFLAGS = $(CXXFLAGS) $(DEBUGFLAGS)
NVCCFLAGS = -arch=sm_86 -O3 -I$(SRCDIR)

# Name of the final executable
TARGET = mshon

# Header dependency groups
TENSOR_HEADERS = $(SRCDIR)/tensor_priv.hpp $(SRCDIR)/tensor.hpp
OPERATION_HEADERS = $(SRCDIR)/operation_priv.hpp $(SRCDIR)/operation.hpp
ARENA_HEADER = $(SRCDIR)/arena.hpp
KERNEL_HELPERS = $(SRCDIR)/cuda_kernels/helpers.hpp

# Operation object files merged into operation.o
OPERATION_OBJS = $(BUILDDIR)/operation_core.o \
	$(BUILDDIR)/op_mmul.o \
	$(BUILDDIR)/op_relu.o \
	$(BUILDDIR)/op_softmax.o \
	$(BUILDDIR)/op_cross_entropy.o \
	$(BUILDDIR)/op_logsoftmax.o \
	$(BUILDDIR)/op_mean.o \
	$(BUILDDIR)/op_addition.o \
	$(BUILDDIR)/op_broadcast.o \
	$(BUILDDIR)/op_embedding.o

# Object files needed to build the target
OBJS = $(BUILDDIR)/main.o \
	$(BUILDDIR)/tensor.o \
	$(BUILDDIR)/operation.o \
	$(BUILDDIR)/arena.o \
	$(BUILDDIR)/mmul.o \
	$(BUILDDIR)/batch_mmul.o \
	$(BUILDDIR)/add.o \
	$(BUILDDIR)/logsoftmax.o \
	$(BUILDDIR)/reduce_add.o \
	$(BUILDDIR)/relu.o

# Default target
.PHONY: all clean
all: $(TARGET)

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) -L/usr/local/cuda/lib64 -lcudart

$(BUILDDIR)/main.o: $(SRCDIR)/main.cpp $(SRCDIR)/mnist_reader.cpp $(TENSOR_HEADERS) $(OPERATION_HEADERS) $(ARENA_HEADER) | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILDDIR)/add.o: $(SRCDIR)/cuda_kernels/add.cu $(KERNEL_HELPERS) | $(BUILDDIR)
	nvcc $(NVCCFLAGS) -c $< -o $@

$(BUILDDIR)/logsoftmax.o: $(SRCDIR)/cuda_kernels/logsoftmax.cu $(KERNEL_HELPERS) $(SRCDIR)/cuda_kernels/logsoftmax.hpp | $(BUILDDIR)
	nvcc $(NVCCFLAGS) -c $< -o $@

$(BUILDDIR)/reduce_add.o: $(SRCDIR)/cuda_kernels/reduce_add.cu $(KERNEL_HELPERS) | $(BUILDDIR)
	nvcc $(NVCCFLAGS) -c $< -o $@

$(BUILDDIR)/relu.o: $(SRCDIR)/cuda_kernels/relu.cu $(KERNEL_HELPERS) | $(BUILDDIR)
	nvcc $(NVCCFLAGS) -c $< -o $@

$(BUILDDIR)/batch_mmul.o: $(SRCDIR)/cuda_kernels/batch_mmul.cu $(KERNEL_HELPERS) | $(BUILDDIR)
	nvcc $(NVCCFLAGS) -c $< -o $@

$(BUILDDIR)/mmul.o: $(SRCDIR)/cuda_kernels/mmul.cu $(KERNEL_HELPERS) | $(BUILDDIR)
	nvcc $(NVCCFLAGS) -c $< -o $@

$(BUILDDIR)/tensor.o: $(SRCDIR)/tensor.cpp $(TENSOR_HEADERS) $(ARENA_HEADER) $(KERNEL_HELPERS) | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILDDIR)/operation.o: $(OPERATION_OBJS) | $(BUILDDIR)
	ld -r -o $@ $(OPERATION_OBJS)

$(BUILDDIR)/operation_core.o: $(SRCDIR)/operation.cpp $(OPERATION_HEADERS) $(TENSOR_HEADERS) $(ARENA_HEADER) $(SRCDIR)/cuda_kernels/add.hpp $(KERNEL_HELPERS) | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILDDIR)/op_mmul.o: $(SRCDIR)/op_mmul.cpp $(OPERATION_HEADERS) $(TENSOR_HEADERS) $(ARENA_HEADER) $(SRCDIR)/cuda_kernels/batch_mmul.hpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILDDIR)/op_relu.o: $(SRCDIR)/op_relu.cpp $(OPERATION_HEADERS) $(TENSOR_HEADERS) $(ARENA_HEADER) $(SRCDIR)/cuda_kernels/relu.hpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILDDIR)/op_softmax.o: $(SRCDIR)/op_softmax.cpp $(OPERATION_HEADERS) $(TENSOR_HEADERS) $(ARENA_HEADER) | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILDDIR)/op_cross_entropy.o: $(SRCDIR)/op_cross_entropy.cpp $(OPERATION_HEADERS) $(TENSOR_HEADERS) $(ARENA_HEADER) | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILDDIR)/op_logsoftmax.o: $(SRCDIR)/op_logsoftmax.cpp $(OPERATION_HEADERS) $(TENSOR_HEADERS) $(ARENA_HEADER) $(SRCDIR)/cuda_kernels/logsoftmax.hpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILDDIR)/op_mean.o: $(SRCDIR)/op_mean.cpp $(OPERATION_HEADERS) $(TENSOR_HEADERS) $(ARENA_HEADER) $(SRCDIR)/cuda_kernels/reduce_add.hpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILDDIR)/op_addition.o: $(SRCDIR)/op_addition.cpp $(OPERATION_HEADERS) $(TENSOR_HEADERS) $(ARENA_HEADER) $(SRCDIR)/cuda_kernels/add.hpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILDDIR)/op_broadcast.o: $(SRCDIR)/op_broadcast.cpp $(OPERATION_HEADERS) $(TENSOR_HEADERS) $(ARENA_HEADER) $(KERNEL_HELPERS) $(SRCDIR)/cuda_kernels/reduce_add.hpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILDDIR)/op_embedding.o: $(SRCDIR)/op_embedding.cpp $(OPERATION_HEADERS) $(TENSOR_HEADERS) $(ARENA_HEADER) | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILDDIR)/arena.o: $(SRCDIR)/arena.cpp $(ARENA_HEADER) $(KERNEL_HELPERS) | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILDDIR) $(TARGET)
