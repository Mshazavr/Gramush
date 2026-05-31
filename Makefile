# Compiler and compilation flags
CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++23 
DEBUGFLAGS = -g -fno-omit-frame-pointer -fsanitize=undefined
ALLFLAGS = $(CXXFLAGS) $(DEBUGFLAGS)

# Name of the final executable
TARGET = mshon

# Object files needed to build the target
OBJS = main.o tensor.o operation.o arena.o

# Default target
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

main.o: main.cpp tensor.hpp operation.hpp arena.hpp
	$(CXX) $(CXXFLAGS) -c main.cpp

tensor.o: tensor.cpp tensor.hpp arena.hpp
	$(CXX) $(CXXFLAGS) -c tensor.cpp

operation.o: operation.cpp operation.hpp tensor.hpp arena.hpp
	$(CXX) $(CXXFLAGS) -c operation.cpp

arena.o: arena.cpp arena.hpp
	$(CXX) $(CXXFLAGS) -c arena.cpp

# 3. Clean step: Remove generated files
clean:
	rm -f $(OBJS) $(TARGET)