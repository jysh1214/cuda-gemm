#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace {
// cuBLAS handle created once and reused across calls
static cublasHandle_t& get_cublas_handle() {
    static cublasHandle_t handle = nullptr;
    if (!handle) {
        cublasCreate(&handle);
    }
    return handle;
}
}  // namespace

// C = A * B, where A is M×K, B is K×N, C is M×N (row-major)
//
// cuBLAS assumes column-major, so we compute:
//   C^T = B^T * A^T
// which is equivalent to C = A * B in row-major.
void cublas_sgemm(const float* A, const float* B, float* C, int M, int N, int K) {
    cublasHandle_t handle = get_cublas_handle();
    float alpha = 1.0f;
    float beta = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N);
}
