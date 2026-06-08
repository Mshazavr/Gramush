# Compiler and compilation flags
CXX = g++
CXXFLAGS = -O3 -Wall -Wextra -std=c++23 -I/usr/local/cuda/include
DEBUGFLAGS = -g -fno-omit-frame-pointer -fsanitize=undefined
ALLFLAGS = $(CXXFLAGS) $(DEBUGFLAGS)

# Name of the final executable
TARGET = mshon

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

operation.o: operation.cpp operation.hpp tensor.hpp arena.hpp $(KERNEL_HEADERS)
	$(CXX) $(CXXFLAGS) -c operation.cpp

arena.o: arena.cpp arena.hpp
	$(CXX) $(CXXFLAGS) -c arena.cpp

# 3. Clean step: Remove generated files
clean:
	rm -f $(OBJS) $(TARGET)