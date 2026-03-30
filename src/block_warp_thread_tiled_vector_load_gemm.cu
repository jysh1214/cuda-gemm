#include <cuda_runtime.h>

namespace {
// block tile: 128×128, each block outputs 128×128 of C
constexpr int BLOCK_TILE_M = 128;
constexpr int BLOCK_TILE_N = 128;
constexpr int BLOCK_TILE_K = 16;
// warp tile: 64×64, each warp outputs 64×64 of C
constexpr int WARP_TILE_M = 64;
constexpr int WARP_TILE_N = 64;
// thread tile: 8×4, each thread computes an 8×4 outer product
constexpr int THREAD_TILE_M = 8;
constexpr int THREAD_TILE_N = 4;
// warp subtile repeats: warp tile is split into WMITER×WNITER subtiles
// WNITER=4 is the design knob, WMITER is derived to cover 64×64 with 32 threads
constexpr int WNITER = 4;
constexpr int WMITER = (WARP_TILE_M * WARP_TILE_N) / (32 * THREAD_TILE_M * THREAD_TILE_N * WNITER);
// warp subtile: 64×16, the region 32 threads cover in one pass (8 rows × 4 cols of 8×4 tiles)
constexpr int WARP_SUB_TILE_M = WARP_TILE_M / WMITER;
constexpr int WARP_SUB_TILE_N = WARP_TILE_N / WNITER;
// 32 threads arranged as WARP_THREAD_ROWS(8) × WARP_THREAD_COLS(4) in each subtile
constexpr int WARP_THREAD_COLS = WARP_SUB_TILE_N / THREAD_TILE_N;
constexpr int WARP_THREAD_ROWS = 32 / WARP_THREAD_COLS;
// 2×2 = 4 warps in the block, 128 threads total
constexpr int NUM_WARPS_M = BLOCK_TILE_M / WARP_TILE_M;
constexpr int NUM_WARPS_N = BLOCK_TILE_N / WARP_TILE_N;
constexpr int BLOCK_THREADS = NUM_WARPS_M * NUM_WARPS_N * 32;
// cooperative loading: 128 threads load 128×16 (A) and 16×128 (B)
constexpr int A_LOADS_PER_THREAD = (BLOCK_TILE_M * BLOCK_TILE_K) / BLOCK_THREADS;
constexpr int B_VEC_LOADS = (BLOCK_TILE_K * BLOCK_TILE_N) / (BLOCK_THREADS * 4);
constexpr int B_VEC_PER_ROW = BLOCK_TILE_N / 4;

static __global__ void __launch_bounds__(BLOCK_THREADS)
    block_warp_thread_tiled_vector_load_gemm_kernel(const float* A, const float* B, float* C, int M,
                                                    int N, int K) {
    int tid = threadIdx.x;

    // warp position in block: 2×2 grid of warps
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int warp_row = warp_id / NUM_WARPS_N;
    int warp_col = warp_id % NUM_WARPS_N;

    // thread position in warp subtile: 8 rows × 4 cols
    int thread_row_in_warp = lane_id / WARP_THREAD_COLS;
    int thread_col_in_warp = lane_id % WARP_THREAD_COLS;

    // block_tile_a is transposed [BK][BM] and padded to avoid bank conflicts
    __shared__ float block_tile_a[BLOCK_TILE_K][BLOCK_TILE_M + 4];
    __shared__ float block_tile_b[BLOCK_TILE_K][BLOCK_TILE_N];

    // each thread accumulates WMITER×WNITER tiles of THREAD_TILE_M×THREAD_TILE_N
    // total per thread: 1×4×8×4 = 128 elements
    float accu[WMITER][WNITER][THREAD_TILE_M][THREAD_TILE_N] = {0.0f};

    // slide BK=16 window along K dimension
    int tile_k_nums = K / BLOCK_TILE_K;
    for (int tile_k_index = 0; tile_k_index < tile_k_nums; tile_k_index++) {
        // 128 threads cooperatively load A(128×16) from global, store transposed into shared
#pragma unroll
        for (int i = 0; i < A_LOADS_PER_THREAD; ++i) {
            int linear = tid + i * BLOCK_THREADS;
            int tile_row = linear / BLOCK_TILE_K;
            int tile_col = linear % BLOCK_TILE_K;
            int global_row = blockIdx.y * BLOCK_TILE_M + tile_row;
            int global_col = tile_k_index * BLOCK_TILE_K + tile_col;
            block_tile_a[tile_col][tile_row] = A[global_row * K + global_col];
        }

        // 128 threads cooperatively load B(16×128) from global using float4
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

        // each k step: load A and B into registers, compute outer products
#pragma unroll
        for (int k = 0; k < BLOCK_TILE_K; ++k) {
            float thread_tile_a[WMITER][THREAD_TILE_M];
            float thread_tile_b[WNITER][THREAD_TILE_N];

            // load thread_tile_a from transposed A (consecutive in shared memory)
            // loaded ONCE per k, reused across all WNITER=4 subtiles
#pragma unroll
            for (int wm = 0; wm < WMITER; ++wm) {
                // warp offset + subtile offset + thread offset
                int row = warp_row * WARP_TILE_M + wm * WARP_SUB_TILE_M +
                          thread_row_in_warp * THREAD_TILE_M;
#pragma unroll
                for (int i = 0; i < THREAD_TILE_M; i += 4) {
                    float4 val = reinterpret_cast<float4*>(&block_tile_a[k][row + i])[0];
                    thread_tile_a[wm][i + 0] = val.x;
                    thread_tile_a[wm][i + 1] = val.y;
                    thread_tile_a[wm][i + 2] = val.z;
                    thread_tile_a[wm][i + 3] = val.w;
                }
            }

            // load thread_tile_b for each of the WNITER=4 subtiles
            // each subtile reads from a different 16-col band of B
#pragma unroll
            for (int wn = 0; wn < WNITER; ++wn) {
                // warp offset + subtile offset + thread offset
                int col = warp_col * WARP_TILE_N + wn * WARP_SUB_TILE_N +
                          thread_col_in_warp * THREAD_TILE_N;
                // TN=4, exactly one float4
                float4 val = reinterpret_cast<float4*>(&block_tile_b[k][col])[0];
                thread_tile_b[wn][0] = val.x;
                thread_tile_b[wn][1] = val.y;
                thread_tile_b[wn][2] = val.z;
                thread_tile_b[wn][3] = val.w;
            }

            // WMITER×WNITER = 1×4 outer products per k
            // thread_tile_a[0] is reused across all 4 WNITER iterations
#pragma unroll
            for (int wm = 0; wm < WMITER; ++wm) {
#pragma unroll
                for (int wn = 0; wn < WNITER; ++wn) {
#pragma unroll
                    for (int r = 0; r < THREAD_TILE_M; ++r) {
#pragma unroll
                        for (int c = 0; c < THREAD_TILE_N; ++c) {
                            accu[wm][wn][r][c] += thread_tile_a[wm][r] * thread_tile_b[wn][c];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    // write accu to C using float4 (TN=4, exactly one float4 per row per subtile)
#pragma unroll
    for (int wm = 0; wm < WMITER; ++wm) {
#pragma unroll
        for (int wn = 0; wn < WNITER; ++wn) {
            // block offset + warp offset + subtile offset + thread offset
            int base_row = blockIdx.y * BLOCK_TILE_M + warp_row * WARP_TILE_M +
                           wm * WARP_SUB_TILE_M + thread_row_in_warp * THREAD_TILE_M;
            int base_col = blockIdx.x * BLOCK_TILE_N + warp_col * WARP_TILE_N +
                           wn * WARP_SUB_TILE_N + thread_col_in_warp * THREAD_TILE_N;
#pragma unroll
            for (int r = 0; r < THREAD_TILE_M; ++r) {
                reinterpret_cast<float4*>(&C[(base_row + r) * N + base_col])[0] = make_float4(
                    accu[wm][wn][r][0], accu[wm][wn][r][1], accu[wm][wn][r][2], accu[wm][wn][r][3]);
            }
        }
    }
}
}  // namespace

void block_warp_thread_tiled_vector_load_gemm(const float* A, const float* B, float* C, int M,
                                              int N, int K) {
    dim3 block(BLOCK_THREADS);
    dim3 grid(N / BLOCK_TILE_N, M / BLOCK_TILE_M);
    block_warp_thread_tiled_vector_load_gemm_kernel<<<grid, block>>>(A, B, C, M, N, K);
}
