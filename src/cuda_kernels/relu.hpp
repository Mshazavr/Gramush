#pragma once 

void cuda_relu(float *A, float *B, int N);
void cuda_relu_backward(float *A, float *AG, float *BG, int N);