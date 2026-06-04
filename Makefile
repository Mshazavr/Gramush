# Compiler and compilation flags
CXX = g++
CXXFLAGS = -O3 -Wall -Wextra -std=c++23 -I/usr/local/cuda/include
DEBUGFLAGS = -g -fno-omit-frame-pointer -fsanitize=undefined
ALLFLAGS = $(CXXFLAGS) $(DEBUGFLAGS)

# Name of the final executable
TARGET = mshon

# Object files needed to build the target
OBJS = main.o tensor.o operation.o arena.o mmul.o batch_mmul.o

# Default target
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) -L/usr/local/cuda/lib64 -lcudart

main.o: main.cpp tensor.hpp operation.hpp arena.hpp
	$(CXX) $(CXXFLAGS) -c main.cpp

batch_mmul.o: cuda_kernels/batch_mmul.cu
	nvcc -O3 -c cuda_kernels/batch_mmul.cu

mmul.o: cuda_kernels/mmul.cu
	nvcc -O3 -c cuda_kernels/mmul.cu

tensor.o: tensor.cpp tensor.hpp arena.hpp
	$(CXX) $(CXXFLAGS) -c tensor.cpp

operation.o: operation.cpp operation.hpp tensor.hpp arena.hpp
	$(CXX) $(CXXFLAGS) -c operation.cpp

arena.o: arena.cpp arena.hpp
	$(CXX) $(CXXFLAGS) -c arena.cpp

# 3. Clean step: Remove generated files
clean:
	rm -f $(OBJS) $(TARGET)