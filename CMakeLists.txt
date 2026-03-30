cmake_minimum_required(VERSION 3.18)
project(cuda_gemm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)
add_executable(gemm
    src/main.cu
    src/naive_gemm.cu
    src/block_tiled_gemm.cu
    src/block_thread_tiled_gemm.cu
    src/block_thread_tiled_vector_load_gemm.cu
    src/block_warp_thread_tiled_vector_load_gemm.cu
    src/cublas_sgemm.cu
)
target_include_directories(gemm PRIVATE include)
target_link_libraries(gemm PRIVATE CUDA::cublas)
