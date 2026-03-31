#include <cuda_runtime.h>

namespace {
constexpr int BLOCK_TILE_M = 64;
constexpr int BLOCK_TILE_N = 64;
constexpr int BLOCK_TILE_K = 8;
constexpr int THREAD_TILE_M = 8;
constexpr int THREAD_TILE_N = 8;
constexpr int BLOCK_THREADS = (BLOCK_TILE_M * BLOCK_TILE_N) / (THREAD_TILE_M * THREAD_TILE_N);
constexpr int A_LOADS_PER_THREAD = (BLOCK_TILE_M * BLOCK_TILE_K) / BLOCK_THREADS;
constexpr int B_VEC_LOADS = (BLOCK_TILE_K * BLOCK_TILE_N) / (BLOCK_THREADS * 4 /* float4 */);
constexpr int B_VEC_PER_ROW = BLOCK_TILE_N / 4 /* float4 */;

static __global__ void block_thread_tiled_vector_load_gemm_kernel(const float* A, const float* B,
                                                                  float* C, int M, int N, int K) {
    int tid = threadIdx.x;

    int thread_start_row = (tid / (BLOCK_TILE_N / THREAD_TILE_N)) * THREAD_TILE_M;
    int thread_start_col = (tid % (BLOCK_TILE_N / THREAD_TILE_N)) * THREAD_TILE_N;

    // transpose tile A for coalescing, padded +4 to:
    // 1. avoid bank conflicts: without padding, stride=64, 64%32=0, all rows hit the same bank.
    //    with +4, stride=68, 68%32=4, consecutive rows shift by 4 banks.
    // 2. preserve float4 alignment: each row starts at offset divisible by 4 floats (16 bytes),
    //    so float4 reads from block_tile_a[k][row] remain aligned.
    __shared__ float block_tile_a[BLOCK_TILE_K][BLOCK_TILE_M + 4];
    __shared__ float block_tile_b[BLOCK_TILE_K][BLOCK_TILE_N];

    float accu[THREAD_TILE_M][THREAD_TILE_N] = {0.0f};

    int tile_k_nums = K / BLOCK_TILE_K;
    for (int tile_k_index = 0; tile_k_index < tile_k_nums; tile_k_index++) {
        // scalar copy tile A to (transpose) shared memory
#pragma unroll
        for (int i = 0; i < A_LOADS_PER_THREAD; ++i) {
            int linear = tid + i * BLOCK_THREADS;
            int tile_row = linear / BLOCK_TILE_K;
            int tile_col = linear % BLOCK_TILE_K;
            int global_row = blockIdx.y * BLOCK_TILE_M + tile_row;
            int global_col = tile_k_index * BLOCK_TILE_K + tile_col;
            block_tile_a[tile_col][tile_row] = A[global_row * K + global_col];
        }

        // vector copy tile B to shared memory
#pragma unroll
        for (int i = 0; i < B_VEC_LOADS; ++i) {
            int linear = tid + i * BLOCK_THREADS;
            int tile_row = linear / B_VEC_PER_ROW;
            int tile_col = (linear % B_VEC_PER_ROW) * 4;
            int global_row = tile_k_index * BLOCK_TILE_K + tile_row;
            int global_col = blockIdx.x * BLOCK_TILE_N + tile_col;
            float4 val = reinterpret_cast<const float4*>(&B[global_row * N + global_col])[0];
            block_tile_b[tile_row][tile_col + 0] = val.x;
            block_tile_b[tile_row][tile_col + 1] = val.y;
            block_tile_b[tile_row][tile_col + 2] = val.z;
            block_tile_b[tile_row][tile_col + 3] = val.w;
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < BLOCK_TILE_K; ++k) {
            float thread_tile_a[THREAD_TILE_M];
            float thread_tile_b[THREAD_TILE_N];

            // float4 read from transposed A
#pragma unroll
            for (int i = 0; i < THREAD_TILE_M; i += 4) {
                float4 val = reinterpret_cast<float4*>(&block_tile_a[k][thread_start_row + i])[0];
                thread_tile_a[i + 0] = val.x;
                thread_tile_a[i + 1] = val.y;
                thread_tile_a[i + 2] = val.z;
                thread_tile_a[i + 3] = val.w;
            }

            // float4 read from B
#pragma unroll
            for (int i = 0; i < THREAD_TILE_N; i += 4) {
                float4 val = reinterpret_cast<float4*>(&block_tile_b[k][thread_start_col + i])[0];
                thread_tile_b[i + 0] = val.x;
                thread_tile_b[i + 1] = val.y;
                thread_tile_b[i + 2] = val.z;
                thread_tile_b[i + 3] = val.w;
            }

#pragma unroll
            for (int r = 0; r < THREAD_TILE_M; ++r) {
#pragma unroll
                for (int c = 0; c < THREAD_TILE_N; ++c) {
                    accu[r][c] += thread_tile_a[r] * thread_tile_b[c];
                }
            }
        }
        __syncthreads();
    }

    // float4 vector store to C
#pragma unroll
    for (int r = 0; r < THREAD_TILE_M; ++r) {
        int global_row = blockIdx.y * BLOCK_TILE_M + thread_start_row + r;
#pragma unroll
        for (int c = 0; c < THREAD_TILE_N; c += 4) {
            int global_col = blockIdx.x * BLOCK_TILE_N + thread_start_col + c;
            reinterpret_cast<float4*>(&C[global_row * N + global_col])[0] =
                make_float4(accu[r][c], accu[r][c + 1], accu[r][c + 2], accu[r][c + 3]);
        }
    }
}
}  // namespace

void block_thread_tiled_vector_load_gemm(const float* A, const float* B, float* C, int M, int N,
                                         int K) {
    dim3 block(BLOCK_THREADS);
    dim3 grid(N / BLOCK_TILE_N, M / BLOCK_TILE_M);
    block_thread_tiled_vector_load_gemm_kernel<<<grid, block>>>(A, B, C, M, N, K);
}
