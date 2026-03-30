#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "block_thread_tiled_gemm.cuh"
#include "block_thread_tiled_vector_load_gemm.cuh"
#include "block_tiled_gemm.cuh"
#include "block_warp_thread_tiled_vector_load_gemm.cuh"
#include "cublas_sgemm.cuh"
#include "naive_gemm.cuh"

struct KernelInfo {
    std::string name;
    std::function<void(const float*, const float*, float*, int, int, int)> fn;
};

std::vector<KernelInfo> kernels = {
    {"naive_gemm", naive_gemm},
    {"block_tiled_gemm", block_tiled_gemm},
    {"block_thread_tiled_gemm", block_thread_tiled_gemm},
    {"block_thread_tiled_vector_load_gemm", block_thread_tiled_vector_load_gemm},
    {"block_warp_thread_tiled_vector_load_gemm", block_warp_thread_tiled_vector_load_gemm},
    {"cublas_sgemm", cublas_sgemm},
};

void cpu_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int p = 0; p < K; ++p) {
                sum += A[i * K + p] * B[p * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

bool check(const float* gpu_C, const float* cpu_C, int M, int N, float rtol = 1e-3f,
           float atol = 1e-3f) {
    for (int i = 0; i < M * N; ++i) {
        float diff = std::fabs(gpu_C[i] - cpu_C[i]);
        float ref = std::fabs(cpu_C[i]);
        if (diff > atol + rtol * ref) {
            std::cerr << "  MISMATCH at index " << i << ": gpu=" << gpu_C[i] << " cpu=" << cpu_C[i]
                      << " diff=" << diff << "\n";
            return false;
        }
    }
    return true;
}

bool test_kernel(float* h_A, float* h_B, float* d_C, int M, int N, int K) {
    size_t size_C = (size_t)M * N * sizeof(float);
    float* h_C_gpu = static_cast<float*>(malloc(size_C));
    float* h_C_cpu = static_cast<float*>(malloc(size_C));
    cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost);
    cpu_gemm(h_A, h_B, h_C_cpu, M, N, K);
    bool result = check(h_C_gpu, h_C_cpu, M, N);
    free(h_C_gpu);
    free(h_C_cpu);
    return result;
}

void fill_random(float* data, int size) {
    for (int i = 0; i < size; ++i) {
        data[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

void benchmark(const std::function<void(const float*, const float*, float*, int, int, int)>& kernel,
               int M, int N, int K, std::ofstream& out, bool test, int warmup_iters = 3,
               int bench_iters = 10) {
    size_t size_A = (size_t)M * K * sizeof(float);
    size_t size_B = (size_t)K * N * sizeof(float);
    size_t size_C = (size_t)M * N * sizeof(float);

    float* h_A = static_cast<float*>(malloc(size_A));
    float* h_B = static_cast<float*>(malloc(size_B));

    fill_random(h_A, M * K);
    fill_random(h_B, K * N);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    for (int i = 0; i < warmup_iters; ++i) {
        kernel(d_A, d_B, d_C, M, N, K);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < bench_iters; ++i) {
        kernel(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= static_cast<float>(bench_iters);

    if (test) {
        if (!test_kernel(h_A, h_B, d_C, M, N, K)) exit(1);
    }

    double gflops = static_cast<double>(2.0 * M * N * K) / static_cast<double>(ms * 1e6);
    out << M << "," << ms << "," << gflops << "\n";
    std::cout << "  Matrix size " << std::left << std::setw(5) << M << std::fixed
              << std::setprecision(4) << ms << " ms  " << std::setprecision(2) << gflops
              << " GFLOPS\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <kernel_index> [--test]\n";
        std::cout << "Available kernels:\n";
        for (size_t i = 0; i < kernels.size(); ++i) {
            std::cout << "  " << i << ": " << kernels[i].name << "\n";
        }
        return 1;
    }

    int idx = std::stoi(argv[1]);
    if (idx < 0 || idx >= static_cast<int>(kernels.size())) {
        std::cerr << "Invalid kernel index " << idx << ". Must be 0.." << kernels.size() - 1
                  << "\n";
        return 1;
    }
    bool test = argc >= 3 && std::string(argv[2]) == "--test";

    auto& kernel = kernels[idx];
    std::string filename = kernel.name + ".csv";
    std::ofstream out(filename);
    out << "size,time_ms,gflops\n";

    std::cout << "Benchmarking: " << kernel.name << "\n";
    for (int size = 256; size <= 8192; size += 256) {
        benchmark(kernel.fn, size, size, size, out, test);
    }

    out.close();
    std::cout << "Results saved to " << filename << "\n";
    return 0;
}
