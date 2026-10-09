# KernelForge

**KernelForge** is an experimental AI-assisted GPU kernel optimization system.
An LLM proposes CUDA kernels; a hardened harness compiles, verifies, benchmarks
and profiles them; the results go back to the LLM for the next attempt.

```text
CUDA kernel -> Compile -> Correctness (many shapes) -> Benchmark vs cuBLAS
     ^                                                        |
     |                                                        v
Optimized kernel  <-  AI analysis  <-  Profile / failure report
```

## Status

Early development. Working today:

- FP32 tiled matmul baseline (`kernels/matmul/baseline.cu`)
- Benchmark runner emitting machine-readable JSON
- Correctness on 7 shapes (incl. non-tile-multiple sizes), double-precision CPU reference
- NaN-poisoned output buffers and a post-benchmark re-check (catches races / unwritten outputs)
- Median/min/mean/stddev timing and GFLOPS, compared against cuBLAS
- `agent/evaluate.py`: builds and runs a candidate in its own process with timeouts

Not yet built: Nsight Compute profiling, Nemotron integration, the optimization loop, dashboard.

## Kernel contract

A candidate is **one `.cu` file** exporting:

```cpp
extern "C" cudaError_t kernelforge_launch(
    const float *A, const float *B, float *C,   // row-major: A MxK, B KxN, C MxN
    int M, int N, int K, cudaStream_t stream);
```

The file owns tile size, block/grid shape, shared memory and coarsening; the
runner knows none of it. Rules the agent must follow: FP32 results within
tolerance, no cuBLAS/cuDNN calls, no host-side compute, launch asynchronously
on `stream`.

## Evaluate a candidate

```bash
python agent/evaluate.py kernels/matmul/baseline.cu            # perf shape 2048^3
python agent/evaluate.py kernels/matmul/baseline.cu 4096 4096 4096
```

Needs `nvcc` on PATH. `KERNELFORGE_ARCH` overrides `-arch` (default `native`;
e.g. `KERNELFORGE_ARCH=sm_90` for a Hopper box).

Output is one JSON object. `status` is one of `ok`, `compile_error`,
`incorrect`, `runtime_error`, `timeout`, `crash`, `setup_error`. On `ok`:

```json
{
  "status": "ok",
  "correctness": {"passed": true, "cases": [ ... ]},
  "kernel": {"median_ms": 0, "min_ms": 0, "mean_ms": 0, "stddev_ms": 0, "gflops": 0},
  "cublas": { ... },
  "percent_of_cublas": 0
}
```

## CMake build (Linux or Windows)

```bash
cmake -S . -B build
cmake --build build --config Release --target matmul_benchmark
./build/matmul_benchmark            # Windows: .\build\Release\matmul_benchmark.exe
```

`-DKERNELFORGE_CUDA_ARCH=86` pins the architecture, `-DKERNEL_SRC=...` selects a candidate.

## Roadmap

- [x] CUDA benchmark infrastructure
- [x] Multi-shape correctness validation
- [x] JSON benchmark results
- [x] Baseline vs cuBLAS
- [ ] GPU profiling (Nsight Compute metrics into the report)
- [ ] Nemotron integration (Nebius Token Factory)
- [ ] Autonomous optimization loop
- [ ] Dashboard
- [ ] Second kernel family (softmax / reduction)
