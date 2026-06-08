#pragma once 

void cuda_reduce_add(float *A, float *B, int M, int N, float scale);
void cuda_reduce_add_backward(float *A, float *B, int M, int N, float scale);