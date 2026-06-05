#pragma once 

void cuda_relu(float *A, int N);
void cuda_relu_backward(float *A, float *BG, int N);