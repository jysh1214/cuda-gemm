# CUDA GEMM

## Kernels

| Index | Name                                     |
|-------|------------------------------------------|
| 0     | naive_gemm                               |
| 1     | block_tiled_gemm                         |
| 2     | block_thread_tiled_gemm                  |
| 3     | block_thread_tiled_vector_load_gemm      |
| 4     | block_warp_thread_tiled_vector_load_gemm |
| 5     | cublas_sgemm                             |

## Build

```bash
cmake -B build -D CMAKE_BUILD_TYPE=Release -D CMAKE_CUDA_ARCHITECTURES=89
cmake --build build
```

## Benchmark

```bash
./build/gemm <kernel_index>
```

## Test

```bash
./build/gemm <kernel_index> --test
```

## Visualize

```bash
draw_plot.py <csv_0> [csv_1] ...
```

Results are saved to `benchmark.png`.

