#pragma once 

void cuda_batch_mmul(float *A, float *B, float *C, int batch_size, int M, int K, int N);