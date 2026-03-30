#include <cuda_runtime.h>

namespace {
static __global__ void naive_gemm_kernel(const float* A, const float* B, float* C, int M, int N,
                                         int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M && col >= N) {
        return;
    }

    float accu = 0.0f;
    for (int k = 0; k < K; ++k) {
        accu += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = accu;
}
}  // namespace

void naive_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(32, 32);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    naive_gemm_kernel<<<grid, block>>>(A, B, C, M, N, K);
}
