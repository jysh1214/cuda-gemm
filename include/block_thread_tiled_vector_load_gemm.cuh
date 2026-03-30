#pragma once

void block_thread_tiled_vector_load_gemm(const float* A, const float* B, float* C, int M, int N,
                                         int K);
