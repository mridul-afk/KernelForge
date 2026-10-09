cmake_minimum_required(VERSION 3.25)

project(KernelForge LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# GPU architecture. "native" targets the GPU in this machine (RTX 3050 = 86).
# Override for another GPU:  -DKERNELFORGE_CUDA_ARCH=90
set(KERNELFORGE_CUDA_ARCH "native" CACHE STRING "CUDA architecture(s) to build for")
set(CMAKE_CUDA_ARCHITECTURES ${KERNELFORGE_CUDA_ARCH})

# Which kernel file to benchmark. The file owns its launch configuration
# (see the kernel contract in kernels/matmul/baseline.cu).
# cmake -B build -DKERNEL_SRC=/path/to/candidate.cu
set(KERNEL_SRC "${CMAKE_SOURCE_DIR}/kernels/matmul/baseline.cu"
  CACHE FILEPATH "Candidate kernel source file")

find_package(CUDAToolkit REQUIRED)

add_executable(matmul_benchmark
  benchmark/runner.cu
  ${KERNEL_SRC}
)
target_link_libraries(matmul_benchmark PRIVATE CUDA::cudart CUDA::cublas)
target_compile_options(matmul_benchmark PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

add_executable(cuda_test tests/cuda_test.cu)
