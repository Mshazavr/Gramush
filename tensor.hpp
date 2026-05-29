#pragma once 

#include <vector>

struct Tensor; 

using TensorHandle = Tensor*;


TensorHandle tensor_zeroes(const std::vector<size_t> &dims, bool learnable);
TensorHandle tensor_random(const std::vector<size_t> &dims, bool learnable);

TensorHandle tensor_from_vector(const std::vector<size_t> &dims, bool learnable, const std::vector<float> &data);
TensorHandle tensor_from_data(const std::vector<size_t> &dims, bool learnable, const float* data);

void tensor_free(TensorHandle tensor);

float tensor_at(TensorHandle tensor, const std::vector<size_t> &indices);

size_t tensor_num_sub_tensors(TensorHandle tensor, size_t num_last_dims);

size_t tensor_size(TensorHandle tensor);