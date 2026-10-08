# KernelForge

**KernelForge** is an experimental AI-assisted GPU kernel optimization system.

The goal is to automatically improve CUDA kernels through an iterative loop of:

```text
CUDA Kernel
    ↓
Compile
    ↓
Correctness Check
    ↓
Benchmark
    ↓
Profile
    ↓
AI Analysis
    ↓
Optimized Kernel
    ↓
Repeat
```

Current Status

🚧 Early development

Currently implemented:

CUDA + CMake project
FP32 tiled matrix multiplication baseline
GPU execution on NVIDIA GPUs
CPU reference implementation
Full correctness validation
CUDA event-based benchmarking
Support for custom matrix dimensions

Current baseline:

GPU: NVIDIA RTX 3050 8GB
CUDA: 13.0
Architecture: sm_86
Tile size: 16 × 16

Example benchmark:

Matrix A: 512 × 1024
Matrix B: 1024 × 512
Matrix C: 512 × 512

Correctness: PASS
Average kernel latency: 1.052181 ms
Project Structure
KernelForge/
├── kernels/
│   └── matmul/
│       └── baseline.cu
├── benchmark/
│   └── runner.cu
├── tests/
│   └── cuda_test.cu
├── agent/
├── experiments/
└── CMakeLists.txt
Build
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release --target matmul_benchmark

Run:

.\build\Release\matmul_benchmark.exe
Roadmap
 CUDA benchmark infrastructure
 Correctness validation
 Baseline performance measurement
 JSON benchmark results
 GPU profiling
 Nemotron integration
 Automatic kernel generation
 Autonomous optimization loop
 User-provided CUDA kernel optimization

KernelForge — An Autonomous GPU Optimization Laboratory

This is the version I'd put in the repo **right now**. It documents the working system without making the project look more complete than it currently is.
