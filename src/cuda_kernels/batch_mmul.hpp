#pragma once 

void cuda_batch_mmul(float *A, bool trans_a, float *B, bool trans_b, float *C, int batch_size, int M, int K, int N, float alpha, float beta);