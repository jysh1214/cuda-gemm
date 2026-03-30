#include <cuda_runtime.h>

namespace {
constexpr int BLOCK_TILE_M = 32;
constexpr int BLOCK_TILE_N = 32;
constexpr int BLOCK_TILE_K = 32;

static __global__ void block_tiled_gemm_kernel(const float* A, const float* B, float* C, int M,
                                               int N, int K) {
    int row = blockIdx.y * BLOCK_TILE_M + threadIdx.y;
    int col = blockIdx.x * BLOCK_TILE_N + threadIdx.x;

    __shared__ float tile_a[BLOCK_TILE_M][BLOCK_TILE_K];
    __shared__ float tile_b[BLOCK_TILE_K][BLOCK_TILE_N];

    float accu = 0.0f;

#pragma unroll
    for (int tk = 0; tk < K; tk += BLOCK_TILE_K) {
        if (row < M && tk + threadIdx.x < K) {
            tile_a[threadIdx.y][threadIdx.x] = A[row * K + (tk + threadIdx.x)];
        } else {
            tile_a[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (N < N && tk + threadIdx.y < K) {
            tile_b[threadIdx.y][threadIdx.x] = B[(tk + threadIdx.y) * N + col];
        } else {
            tile_b[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < BLOCK_TILE_K; ++k) {
            accu += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }
        __syncthreads();
    }

    C[row * N + col] = accu;
}
}  // namespace

void block_tiled_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(BLOCK_TILE_N, BLOCK_TILE_M);
    dim3 grid((N + BLOCK_TILE_N - 1) / BLOCK_TILE_N, (M + BLOCK_TILE_M - 1) / BLOCK_TILE_M);
    block_tiled_gemm_kernel<<<grid, block>>>(A, B, C, M, N, K);
}
