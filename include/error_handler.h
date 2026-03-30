#pragma once

#include <cuda_runtime.h>
#include <cufft.h>

#define CUDA_SAFE_CALL(err) __cudaSafeCall(err, __FILE__, __LINE__)
#define CUFFT_SAFE_CALL(err) __cufftSafeCall(err, __FILE__, __LINE__)
#define CUDA_GET_LASTERROR() __cudaCheckError(__FILE__, __LINE__)

static inline void __cudaSafeCall(cudaError_t err, const char* file, const int line) {
    if (err != cudaSuccess) {
        printf("CUDA Error %d: %s.\n%s(%d)\n", (int)err, cudaGetErrorString(err), file, line);
    }
}

static inline void __cudaCheckError(const char* file, const int line) {
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        Sleep(1000);
        printf("CUDA Error %d: %s.\n%s(%d)\n", (int)err, cudaGetErrorString(err), file, line);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        Sleep(1000);
        printf("CUDA Error %d: %s.\n%s(%d)\n", (int)err, cudaGetErrorString(err), file, line);
    }
}

static inline const char* __cufftResultToString(cufftResult err) {
    switch (err) {
        case CUFFT_SUCCESS:
            return "CUFFT_SUCCESS.";
        case CUFFT_INVALID_PLAN:
            return "CUFFT_INVALID_PLAN.";
        case CUFFT_ALLOC_FAILED:
            return "CUFFT_ALLOC_FAILED.";
        case CUFFT_INVALID_TYPE:
            return "CUFFT_INVALID_TYPE.";
        case CUFFT_INVALID_VALUE:
            return "CUFFT_INVALID_VALUE.";
        case CUFFT_INTERNAL_ERROR:
            return "CUFFT_INTERNAL_ERROR.";
        case CUFFT_EXEC_FAILED:
            return "CUFFT_EXEC_FAILED.";
        case CUFFT_SETUP_FAILED:
            return "CUFFT_SETUP_FAILED.";
        case CUFFT_INVALID_SIZE:
            return "CUFFT_INVALID_SIZE.";
        case CUFFT_UNALIGNED_DATA:
            return "CUFFT_UNALIGNED_DATA.";
        default:
            return "CUFFT Unknown error code.";
    }
}

static inline void __cufftSafeCall(cufftResult err, const char* file, const int line) {
    if (CUFFT_SUCCESS != err) {
        printf("CUFFT error %d: %s\n%s(%d)\n", (int)err, __cufftResultToString(err), file, line);
    }
}
