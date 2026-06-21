# Compiler and compilation flags
CXX = g++
CXXFLAGS = -O3 -Wall -Wextra -std=c++23 -I/usr/local/cuda/include
DEBUGFLAGS = -g -fno-omit-frame-pointer -fsanitize=undefined
ALLFLAGS = $(CXXFLAGS) $(DEBUGFLAGS)

# Name of the final executable
TARGET = mshon

# Operation object files merged into operation.o
OPERATION_OBJS = operation_core.o op_mmul.o op_relu.o op_softmax.o op_cross_entropy.o op_logsoftmax.o op_mean.o op_addition.o op_broadcast.o op_embedding.o

# Object files needed to build the target
OBJS = main.o tensor.o operation.o arena.o mmul.o batch_mmul.o add.o logsoftmax.o reduce_add.o relu.o

# Kernel headers 
KERNEL_HEADERS = cuda_kernels/add.hpp cuda_kernels/batch_mmul.hpp cuda_kernels/mmul.hpp cuda_kernels/logsoftmax.hpp cuda_kernels/reduce_add.hpp cuda_kernels/relu.hpp

# Default target
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) -L/usr/local/cuda/lib64 -lcudart

main.o: main.cpp tensor.hpp operation.hpp arena.hpp
	$(CXX) $(CXXFLAGS) -c main.cpp

add.o: cuda_kernels/add.cu
	nvcc -arch=sm_86 -O3 -c cuda_kernels/add.cu 

logsoftmax.o: cuda_kernels/logsoftmax.cu
	nvcc -arch=sm_86 -O3 -c cuda_kernels/logsoftmax.cu  

reduce_add.o: cuda_kernels/reduce_add.cu
	nvcc -arch=sm_86 -O3 -c cuda_kernels/reduce_add.cu 

relu.o: cuda_kernels/relu.cu
	nvcc -arch=sm_86 -O3 -c cuda_kernels/relu.cu 

batch_mmul.o: cuda_kernels/batch_mmul.cu
	nvcc -arch=sm_86 -O3 -c cuda_kernels/batch_mmul.cu

mmul.o: cuda_kernels/mmul.cu
	nvcc -arch=sm_86 -O3 -c cuda_kernels/mmul.cu

tensor.o: tensor.cpp tensor.hpp arena.hpp
	$(CXX) $(CXXFLAGS) -c tensor.cpp

operation.o: $(OPERATION_OBJS)
	ld -r -o operation.o $(OPERATION_OBJS)

operation_core.o: operation.cpp operation.hpp operation_priv.hpp tensor.hpp arena.hpp $(KERNEL_HEADERS)
	$(CXX) $(CXXFLAGS) -c operation.cpp -o operation_core.o

op_mmul.o: op_mmul.cpp operation.hpp operation_priv.hpp tensor.hpp arena.hpp cuda_kernels/batch_mmul.hpp
	$(CXX) $(CXXFLAGS) -c op_mmul.cpp

op_relu.o: op_relu.cpp operation.hpp operation_priv.hpp tensor.hpp arena.hpp cuda_kernels/relu.hpp
	$(CXX) $(CXXFLAGS) -c op_relu.cpp

op_softmax.o: op_softmax.cpp operation.hpp operation_priv.hpp tensor.hpp arena.hpp
	$(CXX) $(CXXFLAGS) -c op_softmax.cpp

op_cross_entropy.o: op_cross_entropy.cpp operation.hpp operation_priv.hpp tensor.hpp arena.hpp
	$(CXX) $(CXXFLAGS) -c op_cross_entropy.cpp

op_logsoftmax.o: op_logsoftmax.cpp operation.hpp operation_priv.hpp tensor.hpp arena.hpp cuda_kernels/logsoftmax.hpp
	$(CXX) $(CXXFLAGS) -c op_logsoftmax.cpp

op_mean.o: op_mean.cpp operation.hpp operation_priv.hpp tensor.hpp arena.hpp cuda_kernels/reduce_add.hpp
	$(CXX) $(CXXFLAGS) -c op_mean.cpp

op_addition.o: op_addition.cpp operation.hpp operation_priv.hpp tensor.hpp arena.hpp cuda_kernels/add.hpp
	$(CXX) $(CXXFLAGS) -c op_addition.cpp

op_broadcast.o: op_broadcast.cpp operation.hpp operation_priv.hpp tensor.hpp arena.hpp cuda_kernels/helpers.hpp cuda_kernels/reduce_add.hpp
	$(CXX) $(CXXFLAGS) -c op_broadcast.cpp

op_embedding.o: op_embedding.cpp operation.hpp operation_priv.hpp tensor.hpp arena.hpp
	$(CXX) $(CXXFLAGS) -c op_embedding.cpp

arena.o: arena.cpp arena.hpp
	$(CXX) $(CXXFLAGS) -c arena.cpp

# 3. Clean step: Remove generated files
clean:
	rm -f $(OBJS) $(OPERATION_OBJS) $(TARGET)
