#include <cuda_runtime.h>

namespace {
constexpr int BLOCK_TILE_M = 64;
constexpr int BLOCK_TILE_N = 64;
constexpr int BLOCK_TILE_K = 8;
constexpr int THREAD_TILE_M = 8;
constexpr int THREAD_TILE_N = 8;
constexpr int BLOCK_THREADS = (BLOCK_TILE_M * BLOCK_TILE_N) / (THREAD_TILE_M * THREAD_TILE_N);
constexpr int A_LOADS_PER_THREAD = (BLOCK_TILE_M * BLOCK_TILE_K) / BLOCK_THREADS;
constexpr int B_LOADS_PER_THREAD = (BLOCK_TILE_K * BLOCK_TILE_N) / BLOCK_THREADS;

static __global__ void block_thread_tiled_gemm_kernel(const float* A, const float* B, float* C,
                                                      int M, int N, int K) {
    int tid = threadIdx.x;

    // set thread's position in block
    // compute where that 8×8 sub-tile lives inside the 64×64 block tile
    int thread_start_row = (tid / (BLOCK_TILE_N / THREAD_TILE_N)) * THREAD_TILE_M;
    int thread_start_col = (tid % (BLOCK_TILE_N / THREAD_TILE_N)) * THREAD_TILE_N;

    // transpose tile A for coalescing, padded to avoid bank conflicts
    __shared__ float block_tile_a[BLOCK_TILE_K][BLOCK_TILE_M + 1];
    __shared__ float block_tile_b[BLOCK_TILE_K][BLOCK_TILE_N];

    float accu[THREAD_TILE_M][THREAD_TILE_N] = {0.0f};

    int tile_k_nums = (K + BLOCK_TILE_K - 1) / BLOCK_TILE_K;
#pragma unroll
    for (int tile_k_index = 0; tile_k_index < tile_k_nums; tile_k_index++) {
        // copy tile A to (transpose) shared memory
#pragma unroll
        for (int i = 0; i < A_LOADS_PER_THREAD; ++i) {
            int linear = i * BLOCK_THREADS + tid;
            int tile_row = linear / BLOCK_TILE_K;
            int tile_col = linear % BLOCK_TILE_K;
            int global_row = blockIdx.y * BLOCK_TILE_M + tile_row;
            int global_col = tile_k_index * BLOCK_TILE_K + tile_col;
            block_tile_a[tile_col][tile_row] =
                (global_row < M && global_col < K) ? A[global_row * K + global_col] : 0.0f;
        }

        // copy tile B to shared memory
#pragma unroll
        for (int i = 0; i < B_LOADS_PER_THREAD; ++i) {
            int linear = i * BLOCK_THREADS + tid;
            int tile_row = linear / BLOCK_TILE_N;
            int tile_col = linear % BLOCK_TILE_N;
            int global_row = tile_k_index * BLOCK_TILE_K + tile_row;
            int global_col = blockIdx.x * BLOCK_TILE_N + tile_col;
            block_tile_b[tile_row][tile_col] =
                (global_row < K && global_col < N) ? B[global_row * N + global_col] : 0.0f;
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < BLOCK_TILE_K; ++k) {
            float thread_tile_a[THREAD_TILE_M];
            float thread_tile_b[THREAD_TILE_N];

#pragma unroll
            for (int i = 0; i < THREAD_TILE_M; ++i) {
                thread_tile_a[i] = block_tile_a[k][thread_start_row + i];
            }

#pragma unroll
            for (int i = 0; i < THREAD_TILE_N; ++i) {
                thread_tile_b[i] = block_tile_b[k][thread_start_col + i];
            }

            // accumulate outer product
#pragma unroll
            for (int r = 0; r < THREAD_TILE_M; ++r) {
                for (int c = 0; c < THREAD_TILE_N; ++c) {
                    accu[r][c] += thread_tile_a[r] * thread_tile_b[c];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < THREAD_TILE_M; ++r) {
        int global_row = (blockIdx.y * BLOCK_TILE_M) + thread_start_row + r;
#pragma unroll
        for (int c = 0; c < THREAD_TILE_N; ++c) {
            int global_col = (blockIdx.x * BLOCK_TILE_N) + thread_start_col + c;
            if (global_row < M && global_col < N) {
                C[global_row * N + global_col] = accu[r][c];
            }
        }
    }
}
}  // namespace

void block_thread_tiled_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(BLOCK_THREADS);
    dim3 grid((N + BLOCK_TILE_N - 1) / BLOCK_TILE_N, (M + BLOCK_TILE_M - 1) / BLOCK_TILE_M);
    block_thread_tiled_gemm_kernel<<<grid, block>>>(A, B, C, M, N, K);
}
