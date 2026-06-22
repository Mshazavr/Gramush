# Compiler and compilation flags
CXX = g++
SRCDIR = src
BUILDDIR = build
CUDA_HOME ?= /usr/local/cuda
CUDA_ARCH ?= sm_86
CXXFLAGS = -O3 -Wall -Wextra -std=c++23 -I$(CUDA_HOME)/include -I$(SRCDIR) -MMD -MP
DEBUGFLAGS = -g -fno-omit-frame-pointer -fsanitize=undefined
ALLFLAGS = $(CXXFLAGS) $(DEBUGFLAGS)
NVCCFLAGS = -arch=$(CUDA_ARCH) -O3 -I$(SRCDIR) -MMD -MP

TARGET = mshon

CPP_SRCS = $(wildcard $(SRCDIR)/*.cpp) $(wildcard $(SRCDIR)/mnist_utils/*.cpp)
CU_SRCS = $(wildcard $(SRCDIR)/cuda_kernels/*.cu)
HEADER_SRCS = $(wildcard $(SRCDIR)/*.hpp) $(wildcard $(SRCDIR)/mnist_utils/*.hpp) $(wildcard $(SRCDIR)/cuda_kernels/*.hpp)

CPP_OBJS = $(patsubst $(SRCDIR)/%.cpp,$(BUILDDIR)/%.o,$(CPP_SRCS))
CU_OBJS = $(patsubst $(SRCDIR)/cuda_kernels/%.cu,$(BUILDDIR)/%.o,$(CU_SRCS))
OBJS = $(CPP_OBJS) $(CU_OBJS)
DEPS = $(OBJS:.o=.d)

TIDY_FLAGS = -std=c++23 -I$(SRCDIR) -isystem $(CUDA_HOME)/include

.PHONY: all clean format format-check tidy
all: $(TARGET)

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

$(TARGET): $(OBJS) | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) -L$(CUDA_HOME)/lib64 -lcudart

$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp | $(BUILDDIR)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILDDIR)/%.o: $(SRCDIR)/cuda_kernels/%.cu | $(BUILDDIR)
	@mkdir -p $(dir $@)
	nvcc $(NVCCFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILDDIR) $(TARGET)

format:
	clang-format -i $(CPP_SRCS) $(CU_SRCS) $(HEADER_SRCS)

format-check:
	clang-format --dry-run --Werror $(CPP_SRCS) $(CU_SRCS) $(HEADER_SRCS)

tidy:
	@command -v clang-tidy >/dev/null || { echo "clang-tidy not found"; exit 1; }
	clang-tidy $(CPP_SRCS) --warnings-as-errors='*' -- $(TIDY_FLAGS)

-include $(DEPS)
