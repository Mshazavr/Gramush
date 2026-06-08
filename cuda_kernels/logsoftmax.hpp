#pragma once 

void cuda_logsoftmax(float *A, float *B, float *C, float *row_exp_sum, float *row_max, int batch_size, int vec_size);
void cuda_logsoftmax_backward(float *A, float *A_grads, float *B, float *C_grads, float *row_exp_sum, float *row_max, int batch_size, int vec_size);